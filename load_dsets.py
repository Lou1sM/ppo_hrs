import os, json
import torch
from datasets import Dataset, load_dataset, load_from_disk
from typing import List


class SummarizationDataset(Dataset):
    def __init__(self, texts: List[str], summaries: List[str], tokenizer, max_length: int = 1024):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        scenes_txts = self.texts[idx]
        summary = self.summaries[idx]
        inputs = [self.tokenizer(st, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt') for st in scenes_txts]
        targets = self.tokenizer(summary, max_length=self.max_length // 4, truncation=True, padding='max_length', return_tensors='pt')
        return {'input_ids': [x['input_ids'].squeeze() for x in inputs], 'attention_mask': [x['attention_mask'].squeeze() for x in inputs], 'target_ids': targets['input_ids'].squeeze(), 'target_attention_mask': targets['attention_mask'].squeeze()}


def load_dset_by_name(dset_name, tokenizer, recompute):
    if dset_name == 'cnn-dailymail':
        return load_cnn_dailymail(tokenizer, recompute)
    elif dset_name == 'moviesumm':
        return load_moviesumm(tokenizer, recompute)

def load_cnn_dailymail(tokenizer, recompute):
    cached_dset_fp = 'cached_tokenized/cnn'

    if not recompute and os.path.exists(cached_dset_fp):
        dset = load_from_disk(cached_dset_fp)
    else:
        dset = load_dataset('cnn_dailymail', '3.0.0', split='train[:1000]')
        if tokenizer:
            dset = dset.map(
                lambda ex: {
                    **tokenizer(
                        ex['article'],
                        #max_length=max_length,
                        truncation=True,
                        padding='max_length'
                    ),
                    **{f'target_{k}': v for k, v in tokenizer(
                        ex['highlights'],
                        #max_length=max_length // 4,
                        truncation=True,
                        padding='max_length'
                    ).items()}
                },
                batched=True
            )
        dset.save_to_disk(cached_dset_fp)
    return dset

def load_moviesumm(tokenizer, recompute):
    transcripts_dir = '../amazon_video/data/transcripts'
    summaries_dir = '../amazon_video/data/summaries'
    cached_dset_fp = 'cached_tokenized/moviesumm'

    if not recompute and os.path.exists(cached_dset_fp):
        dset = load_from_disk(cached_dset_fp)
    else:
        data = []
        for fname in os.listdir(transcripts_dir):
            if not fname.endswith('.json'):
                continue
            movie_name = os.path.splitext(fname)[0]
            t_fp = os.path.join(transcripts_dir, fname)
            s_fp = os.path.join(summaries_dir, f'{movie_name}.json')
            if not os.path.exists(s_fp):
                continue

            with open(t_fp) as f:
                transcript = json.load(f).get('Transcript', '')
            with open(s_fp) as f:
                summary = json.load(f).get('moviesumm', '')

            scenes = '\n'.join(transcript).split('[SCENE_BREAK]')
            data.append({'input': scenes, 'summary': summary})

        dset = Dataset.from_list(data)

        if tokenizer:
            def tok_fn(ex):
                inputs = tokenizer(
                    ex['input'],
                    max_length=2048,
                    truncation=True,
                    padding='max_length'
                )
                targets = tokenizer(
                    ex['summary'],
                    max_length=512,
                    truncation=True,
                    padding='max_length'
                )
                return {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'target_ids': targets['input_ids'],
                    'target_attention_mask': targets['attention_mask']
                }
            dset = dset.map(tok_fn, batched=False)

        dset.save_to_disk(cached_dset_fp)
    return dset

def collate_fn(batch, is_test=False):
    # batch: list of examples (dicts)
    # each example's "input_ids" is a list of lists of ints (scenes × tokens)
    input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]  # list of (scenes, tokens) tensors
    attention_mask = [torch.tensor(ex["attention_mask"]) for ex in batch]
    target_ids = [torch.tensor(ex["target_ids"]) for ex in batch]
    target_attention_mask = [torch.tensor(ex["target_attention_mask"]) for ex in batch]

    input_ids = torch.stack(input_ids)           # (batch, scenes, tokens)
    attention_mask = torch.stack(attention_mask) # (batch, scenes, tokens)
    target_ids = torch.stack(target_ids)         # (batch, target_len)
    target_attention_mask = torch.stack(target_attention_mask)  # (batch, target_len)

    if is_test:
        input_ids, attention_mask = input_ids[:, :5, :20], attention_mask[:, :5, :20]
        target_ids, target_attention_mask = target_ids[:, :20], target_attention_mask[:, :20]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_attention_mask,
    }
