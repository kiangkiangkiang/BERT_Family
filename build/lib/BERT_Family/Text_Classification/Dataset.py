from torch.utils.data import Dataset
from typing import Optional, Iterable, Any
import pandas as pd
import torch

class ClassificationDataset(Dataset):
    def __init__(
        self, 
        raw_data: pd.DataFrame, 
        raw_target: Optional[Iterable]=None, 
        tokenizer: Any=None, 
        max_length: int=100
        ):

        super().__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        self.raw_data, self.raw_target = raw_data, raw_target
        self.max_length = max_length
        assert self.raw_data.shape[1] <= 2, "Only accept one or two sequences as the input argument."
        if raw_target is not None:
            self.target_key2value = {}
            self.target_value2key = {}
            for i, ele in enumerate(pd.unique(raw_target)):
                self.target_key2value[ele] = i
                self.target_value2key[i] = ele


    def __len__(self):
        return self.raw_data.shape[0]


    def __getitem__(self, idx):
        if self.raw_data.shape[1] == 2:
            result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], 
                                                self.raw_data.iloc[idx, 1], 
                                                padding="max_length", 
                                                max_length=self.max_length, 
                                                truncation=True, 
                                                return_tensors='pt')
        else:
            result = self.tokenizer.encode_plus(self.raw_data.iloc[idx, 0], 
                                                padding="max_length", 
                                                max_length=self.max_length, 
                                                truncation=True, 
                                                return_tensors='pt')

        return result, torch.tensor(self.target_key2value[self.raw_target[idx]]) if self.raw_target is not None else result

