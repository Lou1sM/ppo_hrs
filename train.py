import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm

#torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    #model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name: str = 'llamafactory/tiny-random-Llama-3'
    #model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 1e-8
    num_epochs: int = 1
    warmup_steps: int = 100
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.02
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #device: str = "cpu"

class SummarizationDataset(Dataset):
    def __init__(self, texts: List[str], summaries: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(summary, max_length=self.max_length // 4, truncation=True, padding="max_length", return_tensors="pt")
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "target_ids": targets["input_ids"].squeeze(), "target_attention_mask": targets["attention_mask"].squeeze()}

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
    def __init__(self, summ_model, prev_summ_model, tokenizer, config: TrainingConfig):
        self.summ_model = summ_model
        self.prev_summ_model = prev_summ_model
        self.tokenizer = tokenizer
        self.config = config
        summ_model_for_val_net = deepcopy(summ_model)
        self.value_net = ValueNetwork(summ_model_for_val_net).to(config.device)
        self.summ_model_opt = torch.optim.AdamW(summ_model.parameters(), lr=config.learning_rate)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=config.learning_rate)
        self.summ_model.to(config.device)
        self.prev_summ_model.to(config.device)
        self.disc_fac = 0.9
        self.max_inter_summ_len = 10
        self.eps = 0.2

    def generate_summary(self, model, input_ids, attention_mask, max_length=100):
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True, pad_token_id=self.tokenizer.pad_token_id, do_sample=True, temperature=0.7)
        return outputs

    def compute_perplexity_reward(self, first_summaries, input_ids, attention_mask):
        rewards = []
        for summary_ids in first_summaries:
            summary_text = self.tokenizer.decode(summary_ids, skip_special_tokens=True)
            summary_inputs = self.tokenizer(summary_text, return_tensors="pt", max_length=self.config.max_length // 4, truncation=True, padding=True).to(self.config.device)
            with torch.no_grad():
                outputs = self.prev_summ_model(**summary_inputs, labels=summary_inputs["input_ids"])
                perplexity = torch.exp(outputs.loss)
                reward = -perplexity.item()
                rewards.append(reward)
        return torch.tensor(rewards, device=self.config.device)

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i]
            else:
                next_value = values[i + 1]
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, device=self.config.device)

    def ppo_update(self, states, actions, rewards, old_log_probs, advantages, values):
        for _ in range(self.config.ppo_epochs):
            outputs = self.summ_model(input_ids=states["input_ids"], attention_mask=states["attention_mask"])
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            current_values = self.value_net(states["input_ids"], states["attention_mask"])
            value_loss = F.mse_loss(current_values, rewards + self.config.gamma * values)
            entropy = -torch.sum(F.softmax(logits, dim=-1) * log_probs, dim=-1).mean()
            entropy_loss = -self.config.entropy_coeff * entropy
            total_loss = policy_loss + self.config.value_coeff * value_loss + entropy_loss
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.summ_model.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

    def train_step(self, batch):
        batch = {k:v.to(self.config.device) for k,v in batch.items()}
        inter_summ_outputs = self.summ_model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=self.max_inter_summ_len, pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True, temperature=1, top_p=1, do_sample=False)
        inter_summ_ids = inter_summ_outputs.sequences[:,-self.max_inter_summ_len:]
        eos_mask = torch.cumsum((inter_summ_ids == self.tokenizer.eos_token_id).long(), dim=1)
        inter_summ_attn_mask = (eos_mask <= 1).long()

        final_summ_input_ids = torch.cat([inter_summ_ids, batch['target_ids']], axis=1)
        final_summ_attn_mask = torch.cat([inter_summ_attn_mask, batch['target_attention_mask']], axis=1)
        final_summ_labels = torch.cat([-100*torch.ones_like(inter_summ_ids), batch['target_ids']], axis=1)
        final_summ_outputs = self.summ_model(input_ids=final_summ_input_ids, attention_mask=final_summ_attn_mask, labels=final_summ_labels)
        logits = final_summ_outputs.logits
        vocab_size = logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, vocab_size), final_summ_labels.view(-1))
        losses = losses.view(final_summ_labels.size())  # shape [batch_size, seq_len]
        per_item_loss = losses.mean(dim=1)  # mean over tokens per example
        final_reward = -per_item_loss.detach()
        final_summ_loss = per_item_loss.mean()

        value_preds = self.value_net(batch['input_ids'], inter_summ_ids) # assume input_ids already contains a prompt telling model to summarise
        #value_loss = (value_preds**2).sum()
        N = inter_summ_attn_mask.sum(axis=1)
        advantages_lst = []
        value_loss = 0
        for bidx in range(self.config.batch_size):
            bfr = final_reward[bidx:bidx+1].detach() # idx like this to keepdim
            bvp = value_preds[bidx, :N[bidx]+1]
            value_targets = self.disc_fac*value_preds[bidx, 1:N[bidx]+1].detach() # shift by 1 because value_preds predicts sth for the bare prompt at start
            badvs = torch.cat([value_targets[:-1] - bvp[:-2], bfr - bvp[-2:-1]])
            advantages_lst.append(badvs.detach())
            value_loss = value_loss + ((badvs[:-1]**2).sum() + (value_targets[-1] - bvp[-2])**2 + (bfr - bvp[-1])**2 ) / N[bidx]
            #value_loss += ((value_targets[-1] - bvp[-2])**2 + (bfr - bvp[-1])**2 ) / N[bidx]
            #value_loss += ((0.5 - bvp[-2])**2 + (0.5 - bvp[-1])**2 )

        value_loss = value_loss / self.config.batch_size
        advantages = torch.stack(advantages_lst)
        # PPO update
        pis_inputs = torch.cat([batch['input_ids'], inter_summ_ids], axis=1)
        pis_attn = torch.cat([batch['attention_mask'], inter_summ_attn_mask], axis=1)
        def probs_from_model(m):
            with torch.no_grad():
                m_outputs = self.prev_summ_model(input_ids=pis_inputs, attention_mask=pis_attn)#, labels=final_summ_labels)
            #curr_logits = torch.stack(inter_summ_outputs.scores, axis=1)
            logits = m_outputs.logits[:, batch['input_ids'].shape[1]:]
            all_probs = F.softmax(logits, dim=-1)
            trans_probs = torch.gather(all_probs, 2, inter_summ_ids[:,:,None]).squeeze(2)
            return trans_probs

        curr_trans_probs = probs_from_model(self.summ_model)
        prev_trans_probs = probs_from_model(self.prev_summ_model)
            #curr_probs = F.softmax(curr_logits, dim=-1)
            #prev_probs = F.softmax(prev_logits, dim=-1)
            #curr_trans_probs = torch.gather(curr_probs, 2, inter_summ_ids[:,:,None]).squeeze(2)
        #prev_trans_probs = torch.gather(prev_probs, 2, inter_summ_ids[:,:,None]).squeeze(2)
        #print(curr_trans_probs)
        #print(prev_trans_probs)
        #x=self.summ_model(input_ids=pis_inputs, attention_mask=pis_attn)
        #curr_probs_other_way = F.softmax(x.logits[:, batch['input_ids'].shape[1]:], dim=-1)
        reward_ratios = curr_trans_probs / prev_trans_probs
        clipped_reward_ratios = torch.clip(reward_ratios, min=1-self.eps, max=1+self.eps)
        ppo_objective = torch.minimum(reward_ratios*advantages, clipped_reward_ratios*advantages)

        mask = inter_summ_attn_mask / inter_summ_attn_mask.sum(axis=1, keepdim=True)
        policy_loss = (-ppo_objective * mask).sum()
        policy_loss = -ppo_objective.mean()
        self.prev_summ_model = deepcopy(self.summ_model) # copy right before updating summ_model
        self.summ_model_opt.zero_grad(); (policy_loss + final_summ_loss).backward();
        torch.nn.utils.clip_grad_norm_(self.summ_model.parameters(), max_norm=1.0)
        self.summ_model_opt.step()
        self.value_optimizer.zero_grad(); value_loss.backward(); self.value_optimizer.step()

        return {"policy_loss": policy_loss, "value_loss": value_loss, "reward": final_reward.mean().item()}

def train_summ_model_ntp(model, tokenizer, train_dset, config: TrainingConfig):
    logger.info("Training Model 2 with Next-Token Prediction...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(train_dset, batch_size=config.batch_size, shuffle=True)
    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Model 2 - Epoch {epoch+1}"):
            input_ids = batch["target_ids"].to(config.device)
            attention_mask = batch["target_attention_mask"].to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            break
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        logger.info(f"Model 2 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    return model

def main():
    config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", llm_int8_enable_fp32_cpu_offload=True)
    #summ_model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=bnb_config, device_map={"": "cpu"})
    #prev_summ_model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=bnb_config, device_map={"": "cpu"})
    summ_model = AutoModelForCausalLM.from_pretrained(config.model_name, load_in_4bit=True).to(config.device) # policy network
    prev_summ_model = deepcopy(summ_model)

    logger.info("Loading dset...")
    if os.path.exists(cached_dset_fp:='cached_cnn_dset'):
        dset = load_from_disk(cached_dset_fp)
    else:
        dset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
        dset.save_to_disk(cached_dset_fp)

    texts = [item["article"] for item in dset]
    summaries = [item["highlights"] for item in dset]
    train_dset = SummarizationDataset(texts, summaries, tokenizer, config.max_length)
    dataloader = DataLoader(train_dset, batch_size=config.batch_size, shuffle=True)

    #summ_model = train_summ_model_ntp(summ_model, tokenizer, train_dset, config)

    logger.info("Training Model 1 with PPO...")
    ppo_trainer = PPOTrainer(summ_model, prev_summ_model, tokenizer, config)
    summ_model.train()
    prev_summ_model.eval()

    for epoch in range(config.num_epochs):
        epoch_losses = []
        epoch_rewards = []
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            #try:
            metrics = ppo_trainer.train_step(batch)
            #epoch_losses.append(metrics["policy_loss"])
            #epoch_rewards.append(metrics["reward"])
            #if batch_idx % 100 == 0:
            #    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Reward: {metrics['reward']:.4f}")
            #except Exception as e:
                #logger.error(f"Error in batch {batch_idx}: {e}")
                #continue
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    summ_model.save_pretrained("./summ_model_ppo")
    prev_summ_model.save_pretrained("./prev_summ_model_ntp")
    tokenizer.save_pretrained("./tokenizer")
    logger.info("Training completed!")

    #demo_text = "The quick brown fox jumps over the lazy dog. This is a simple example text for demonstration purposes."
    #inputs = tokenizer(demo_text, return_tensors="pt", max_length=config.max_length, truncation=True)
    #with torch.no_grad():
    #    summary1_ids = summ_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
    #    summary1 = tokenizer.decode(summary1_ids[0], skip_special_tokens=True)
    #summary1_inputs = tokenizer(summary1, return_tensors="pt", max_length=config.max_length, truncation=True)
    #with torch.no_grad():
    #    summary2_ids = prev_summ_model.generate(**summary1_inputs, max_length=30, num_beams=4, early_stopping=True)
    #    summary2 = tokenizer.decode(summary2_ids[0], skip_special_tokens=True)

    #print(f"\nOriginal text: {demo_text}")
    #print(f"First summary (Model 1): {summary1}")
    #print(f"Final summary (Model 2): {summary2}")

if __name__ == "__main__":
    main()
