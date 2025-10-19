import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from load_dsets import load_dset_by_name, collate_fn
import numpy as np
import logging
from dataclasses import dataclass
from tqdm import tqdm

#torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    model_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    #model_name: str = 'llamafactory/tiny-random-Llama-3'
    #model_name: str = 'meta-llama/Llama-3.1-8B-Instruct'
    dset_name: str = 'moviesumm'
    max_length: int = 512
    batch_size: int = 1
    learning_rate: float = 1e-6
    num_epochs: int = 10
    warmup_steps: int = 100
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.02
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class ValueNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(model.config.hidden_size, 1, dtype=model.dtype)

    def forward(self, input_ids_, inter_summ_ids):
        val_input_ids = torch.cat([input_ids_, inter_summ_ids], axis=1)
        outputs = self.model(input_ids=val_input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:,input_ids_.shape[1]-1:]
        value = self.value_head(hidden_states)
        return value.squeeze(-1)

class PPOTrainer:
    def __init__(self, summ_model, prev_summ_model, tokenizer, args):
        self.summ_model = summ_model
        self.prev_summ_model = prev_summ_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        summ_model_for_val_net = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_4bit=True).to(args.device) # policy networkdeepcopy(summ_model)
        self.value_net = ValueNetwork(summ_model_for_val_net).to(args.device)
        self.summ_model_opt = torch.optim.AdamW(summ_model.parameters(), lr=args.learning_rate, eps=1)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=args.learning_rate, eps=1)
        self.summ_model.to(args.device)
        self.prev_summ_model.to(args.device)
        self.disc_fac = 0.9
        self.max_inter_summ_len = args.max_inter_summ_len
        self.eps = 0.2

    def train_step(self, batch):
        batch = {k:v.to(self.device) for k,v in batch.items()}
        inter_summ_outputs = self.summ_model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=self.max_inter_summ_len, pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True, temperature=1, top_p=1, do_sample=False)
        inter_summ_ids = torch.tensor(inter_summ_outputs.sequences)[:,-self.max_inter_summ_len:]
        eos_mask = torch.cumsum((inter_summ_ids == self.tokenizer.eos_token_id).long(), dim=1)
        inter_summ_attn_mask = (eos_mask <= 1).long()

        final_summ_input_ids = torch.cat([inter_summ_ids.flatten(), batch['target_ids'].squeeze(0)])
        final_summ_attn_mask = torch.cat([inter_summ_attn_mask.flatten(), batch['target_attention_mask'].squeeze(0)])
        final_summ_labels = torch.cat([-100*torch.ones_like(inter_summ_ids.flatten()), batch['target_ids'].squeeze(0)])
        final_summ_outputs = self.summ_model(input_ids=final_summ_input_ids.unsqueeze(0), attention_mask=final_summ_attn_mask.unsqueeze(0), labels=final_summ_labels.unsqueeze(0))
        logits = final_summ_outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits.permute(0,2,1), final_summ_labels.unsqueeze(0))
        per_item_loss = losses.mean(dim=1)  # mean over tokens per example
        final_reward = -per_item_loss.detach()
        final_summ_loss = per_item_loss.mean()

        value_preds = self.value_net(batch['input_ids'], inter_summ_ids) # assume input_ids already contains a prompt telling model to summarise
        N = inter_summ_attn_mask.sum(axis=1)
        advantages_lst = []
        value_loss = 0
        for bidx in range(self.batch_size):
            bfr = final_reward[bidx:bidx+1].detach() # idx this way in order to keepdim
            bvp = value_preds[bidx, :N[bidx]+1]
            value_targets = self.disc_fac*value_preds[bidx, 1:N[bidx]+1].detach() # shift by 1 because value_preds predicts sth for the bare prompt at start
            badvs = torch.cat([value_targets[:-1] - bvp[:-2], bfr - bvp[-2:-1]])
            if len(badvs) != self.max_inter_summ_len:
                breakpoint()
            badvs = torch.cat([badvs, torch.zeros(self.max_inter_summ_len - len(badvs), device=self.summ_model.device)])
            advantages_lst.append(badvs.detach())
            value_loss = value_loss + ((badvs[:-1]**2).sum() + (value_targets[-1] - bvp[-2])**2 + (bfr - bvp[-1])**2 ) / N[bidx]
            if value_loss.isnan():
                breakpoint()

        #value_loss = 1e-5*value_loss / self.batch_size
        value_loss = value_loss / self.batch_size
        advantages = torch.stack(advantages_lst)
        # PPO update
        pis_inputs = torch.cat([batch['input_ids'], inter_summ_ids], axis=1)
        pis_attn = torch.cat([batch['attention_mask'], inter_summ_attn_mask], axis=1)
        def probs_from_model(m):
            with torch.no_grad():
                m_outputs = m(input_ids=pis_inputs, attention_mask=pis_attn)#, labels=final_summ_labels)
            logits = m_outputs.logits[:, batch['input_ids'].shape[1]:]
            all_probs = F.softmax(logits, dim=-1)
            trans_probs = torch.gather(all_probs, 2, inter_summ_ids[:,:,None]).squeeze(2)
            return trans_probs

        curr_trans_probs = probs_from_model(self.summ_model)
        prev_trans_probs = probs_from_model(self.prev_summ_model)
        reward_ratios = curr_trans_probs / (prev_trans_probs+1e-5)
        clipped_reward_ratios = torch.clip(reward_ratios, min=1-self.eps, max=1+self.eps)
        ppo_objective = torch.minimum(reward_ratios*advantages, clipped_reward_ratios*advantages)

        mask = inter_summ_attn_mask / inter_summ_attn_mask.sum(axis=1, keepdim=True)
        policy_loss = (-ppo_objective * mask).sum()
        if policy_loss.isnan():
            breakpoint()
        (policy_loss + final_summ_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.summ_model.parameters(), max_norm=1.0)
        self.summ_model_opt.step()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        scores = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'reward': final_reward.mean().item()}
        after_nans = [n for n,x in self.summ_model.named_parameters() if x.isnan().any()]
        if len(after_nans) > 0:
            print(f'nan params in summ_model: {after_nans}')
            breakpoint()
        return scores

    def generate_example(self, example_input):
        inter_summ_outputs = self.summ_model.generate(input_ids=torch.tensor(example_input['input_ids']).to(self.summ_model.device)[:3,-1000:], max_new_tokens=self.max_inter_summ_len, min_new_tokens=self.max_inter_summ_len-20, pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True, temperature=1, top_p=1, do_sample=False)
        breakpoint()
        inter_summ_ids = inter_summ_outputs.sequences[0,-self.max_inter_summ_len:]
        final_summ_input_ids = torch.cat([torch.tensor(self.tokenizer('Summarise the following text: ').input_ids, device=self.summ_model.device), inter_summ_ids], axis=0)
        final_summ_genned_ids = self.summ_model.generate(input_ids=final_summ_input_ids.unsqueeze(0))
        final_text = self.tokenizer.decode(final_summ_genned_ids[0], skip_special_tokens=True)
        inp_text = ' '.join(self.tokenizer.batch_decode(torch.tensor(example_input['input_ids']), skip_special_tokens=True))
        inter_text = ' '.join(self.tokenizer.batch_decode(inter_summ_ids, skip_special_tokens=True))
        print(f'Input: {inp_text}\nInter: {inter_text}\nOutput: {final_text}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset-prefix', type=str, default='.')
    parser.add_argument('--model-save-dir', type=str, default='.')
    parser.add_argument('--is-test', '-t', action='store_true')
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--dset_name', type=str, default='moviesumm')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max-inter-summ-len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--ppo_clip_ratio', type=float, default=0.02)
    parser.add_argument('--value_coeff', type=float, default=0.5)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ARGS = parser.parse_args()

    #config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info('Loading dset...')
    train_dset = load_dset_by_name(ARGS.dset_name, tokenizer, ARGS.dset_prefix, recompute=False)

    truncated_collate_fn = partial(collate_fn, is_test=True)
    dataloader = DataLoader(train_dset, collate_fn=truncated_collate_fn, batch_size=ARGS.batch_size, shuffle=True)

    logger.info('Training Model 1 with PPO...')
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    summ_model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', quantization_config=bnb_config, device_map="auto")
    #prev_summ_model = AutoModelForCausalLM.from_pretrained(ARGS.model_name, load_in_4bit=True).to(ARGS.device) # policy networkdeepcopy(summ_model)
    prev_summ_model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', quantization_config=bnb_config, device_map="auto")

    ppo_trainer = PPOTrainer(summ_model, prev_summ_model, tokenizer, ARGS)
    summ_model.train()
    prev_summ_model.eval()

    for epoch in range(ARGS.num_epochs):
        epoch_plosses = []
        epoch_vlosses = []
        epoch_rewards = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            metrics = ppo_trainer.train_step(batch)
            epoch_plosses.append(metrics['policy_loss'])
            epoch_vlosses.append(metrics['value_loss'])
            epoch_rewards.append(metrics['reward'])

            avg_ploss = np.mean(epoch_plosses)
            avg_vloss = np.mean(epoch_vlosses)
            avg_reward = np.mean(epoch_rewards)
            if np.isnan(avg_ploss):
                breakpoint()
            pbar.set_description(f'Epoch {epoch+1} | Policy loss: {avg_ploss:.4f} | Value loss: {avg_vloss:.4f} | Reward: {avg_reward:.4f}')

            if ((batch_idx+1) % 100 == 0) or ARGS.is_test:
                ppo_trainer.generate_example(np.random.choice(train_dset))

            if ARGS.is_test:
                break

    summ_model.save_pretrained(os.path.join(ARGS.model_save_dir, 'summ_model_ppo'))
    prev_summ_model.save_pretrained(os.path.join(ARGS.model_save_dir, 'prev_summ_model_ntp'))
    tokenizer.save_pretrained(os.path.join(ARGS.model_save_dir, 'tokenizer'))
    logger.info('Training completed!')


if __name__ == '__main__':
    main()
