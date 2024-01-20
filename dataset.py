import re
import math
import string
import os
import torch
import pytorch_lightning as pl
import pickle

from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from itertools import chain
from transformers import default_data_collator


def prepare_data(path='./data/wikitext2'):
    if (path is not None) and (not os.path.isdir(path)):
        print("Downloading and processing dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        dataset.save_to_disk(path)
    else:
        print("Dataset already downloaded and processed")
        dataset = load_from_disk(path)
    return dataset


def preprocess_datasets(
        raw_dataset, 
        tokenizer, 
        block_size=64, 
        overwrite_cache=False, 
        preprocessing_num_workers=4):
    column_names = raw_dataset['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        keep_in_memory=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        keep_in_memory=True
    )
    return dataset


class MyDataLoader(pl.LightningDataModule):

    def __init__(
            self, dataset_name, workers, train_dataset, val_dataset,
            test_dataset, batch_size):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.batch_size = batch_size
        self.num_workers = workers

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

class WikiText2Dataset(Dataset):
    path = './data/wikitext2'
    def __init__(
            self, 
            dataset,
            partition,
            tokenizer=None, 
            max_token_count=512):
        self.setup_tokenizer(tokenizer, max_token_count)
        self.dataset = dataset[partition]

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=torch.tensor(data_row['input_ids']),
            attention_mask=torch.tensor(data_row['attention_mask']),
            labels=torch.tensor(data_row['labels']))


class GeneratedDataset(Dataset):
    path = './data/wikitext2'

    def __init__(
            self,
            my_file,
            partition,
            tokenizer=None,
            max_token_count=512):
        with open(my_file, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=data_row['input_ids'],
            attention_mask=data_row['attention_mask'],
            labels=data_row['labels'])
