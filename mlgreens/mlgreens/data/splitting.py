"""Data handling utilities for MLGreens
"""

import torch
import numpy as np
from numpy import random

class DataSubset(torch.utils.data.Dataset):
    """A subset of a larger dataset"""

    def __init__(self, parent_set, indices):
        self.parent_set = parent_set
        self.indices = indices

    def __len__(self):
        return(len(self.indices))
    
    def __getitem__(self, index):
        parent_index = self.indices[index]
        return self.parent_set[parent_index]

def split_data(data, split, seed=None):
    """Splits a dataset in two and returns the subsets

    Args:
      split: splitting fraction
      seed: fandom seed (default None)
    """

    if seed:
        random.seed(seed)

    n_data = len(data)
    n_set1 = int(split * n_data)
    
    set1_ids = set(random.choice(range(n_data), size=n_set1))
    subset1 = DataSubset(data, list(set1_ids))

    set2_ids = set(range(n_data)) - set1_ids
    subset2= DataSubset(data, list(set2_ids))

    return subset1, subset2
