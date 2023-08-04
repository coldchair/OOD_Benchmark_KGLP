import random
import torch
from .dataset import KGDataset

class KGDataloader(object):
    def __init__(
        self, 
        dataset: KGDataset, 
        eta: int, 
        batch_size: int = None, 
        batch_count: int = None,
        shuffle: bool = True,
        mode: str = "ht",
    ):
        if (batch_size is None and batch_count is None) or (batch_size is not None and batch_count is not None):
            raise ValueError("batch_size and batch_count should be given just one of them.")
        if batch_size is None:
            batch_size = max(len(dataset) // batch_count, 1)
        self.dataset = dataset
        self.eta = eta
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def _generate_negative_triples(self, batch: torch.IntTensor):
        if self.eta == 0:
            return None

        mode = self.mode

        eta = self.eta
        dataset = torch.reshape(
            torch.tile(torch.reshape(batch, [-1]), [eta]),
            [batch.shape[0] * eta, 3]
        )
        if mode == "ht": 
            keep_head_mask = torch.randint(
                0, 2, [dataset.shape[0]], dtype=torch.bool
            )
            
            keep_tail_mask = ~keep_head_mask

        elif mode == "h":
            keep_head_mask = torch.zeros((dataset.shape[0],), dtype=torch.bool)
            keep_tail_mask = ~keep_head_mask

        elif mode == "t":
            keep_head_mask = torch.ones((dataset.shape[0],), dtype=torch.bool)
            keep_tail_mask = ~keep_head_mask

        else:
            raise ValueError(f"mode {mode} negative sampling does not support.")

        keep_head_mask = keep_head_mask.int()
        keep_tail_mask = keep_tail_mask.int()

        replacements = torch.randint(
            0, self.dataset.num_entities, [dataset.shape[0]]
        )

        head = keep_head_mask * dataset[:, 0] + keep_tail_mask * replacements
        tail = keep_tail_mask * dataset[:, 2] + keep_head_mask * replacements
        rela = dataset[:, 1]
        ret = torch.stack([head, rela, tail]).transpose(0, 1) # (batch_size * eta, 3)
        ret = ret.view(eta, batch.shape[0], 3).transpose(0, 1).contiguous()

        return ret       


    def __iter__(self):
        _len = len(self.dataset)
        index = list(range(_len))
        if self.shuffle:
            random.shuffle(index)
        cur = 0
        batch = []
        while cur < _len:
            batch.append(torch.tensor(self.dataset[index[cur]]))
            if len(batch) == self.batch_size:
                batch = torch.stack(batch, dim=0)
                yield {
                    "positives": batch,
                    "negatives": self._generate_negative_triples(batch)
                }
                batch = []

            cur += 1
        
        