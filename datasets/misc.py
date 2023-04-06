from typing import Dict, List
import torch
from einops import rearrange

def collate_fn_general(batch: List) -> Dict:
    """ General collate function used for dataloader.
    """
    batch_data = batch
    return batch_data

