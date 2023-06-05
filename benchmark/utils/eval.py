import numpy as np
import torch
from enum import Enum
from pykeen.models import ERModel
from typing import Tuple, Optional
from collections import defaultdict
import tqdm

from dataset import KGDataset

class RankingStrategy(Enum):
    WORST = 0
    MEAN = 1
    BEST = 2

def eval(
    model: ERModel,
    bg_dataset: KGDataset,
    test_dataset: KGDataset,
    batch_size: int,
    filter_datasets: Optional[Tuple[KGDataset]] = None,
    ranking_strategy: RankingStrategy = RankingStrategy.WORST,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # head predict
    head_rank_result_map = defaultdict(lambda: None)
    rt_to_triples = defaultdict(list)
    for triple in test_dataset:
        rt = (triple[1], triple[2])
        rt_to_triples[rt].append(triple[0])
    
    rt_to_fileters = defaultdict(list)
    if filter_datasets is not None:
        for dataset in filter_datasets:
            for triple in dataset:
                rt = (triple[1], triple[2])
                rt_to_fileters[rt].append(triple[0])
    
    for rt, h_list in tqdm.tqdm(rt_to_triples.items(), desc="Eval Head Prediction"):
        h_scores: torch.FloatTensor = model.score_h(
            torch.tensor([[rt[0], rt[1]]], dtype=torch.int32, device=device),
            slice_size=batch_size
        ).squeeze()
        mask_index = rt_to_fileters[rt]
        mask_index = list(set(mask_index))
        mask_index.sort()
        mask = torch.ones_like(h_scores, dtype=torch.bool)
        mask[mask_index] = False
        h_list = list(set(h_list))
        for h in h_list:
            score = h_scores[h].item()
            _mask = mask
            _mask[h] = False
            worst = ((h_scores >= score) & _mask).sum().item() + 1
            best = ((h_scores > score) & _mask).sum().item() + 1
            mean = (best + worst) / 2
            _ret = [worst, mean, best]
            head_rank_result_map[(h, rt[0], rt[1])] = _ret[ranking_strategy.value]
        
        
    
    # tail predict
    tail_rank_result_map = defaultdict(lambda: None)
    hr_to_triples = defaultdict(list)
    for triple in test_dataset:
        hr = (triple[0], triple[1])
        hr_to_triples[hr].append(triple[2])
    
    hr_to_fileters = defaultdict(list)
    if filter_datasets is not None:
        for dataset in filter_datasets:
            for triple in dataset:
                hr = (triple[0], triple[1])
                hr_to_fileters[hr].append(triple[2])
    
    for hr, t_list in tqdm.tqdm(hr_to_triples.items(), desc="Eval Tail Prediction"):
        t_scores: torch.FloatTensor = model.score_t(
            torch.tensor([[hr[0], hr[1]]], dtype=torch.int32),
            slice_size=batch_size
        ).squeeze()
        mask_index = hr_to_fileters[hr]
        mask_index = list(set(mask_index))
        mask_index.sort()
        mask = torch.ones_like(t_scores, dtype=torch.bool)
        mask[mask_index] = False
        t_list = list(set(t_list))
        for t in t_list:
            score = t_scores[t].item()
            _mask = mask
            _mask[t] = False
            worst = ((t_scores >= score) & _mask).sum().item() + 1
            best = ((t_scores > score) & _mask).sum().item() + 1
            mean = (best + worst) / 2
            _ret = [worst, mean, best]
            tail_rank_result_map[(hr[0], hr[1], t)] = _ret[ranking_strategy.value]

    ret = []
    for triple in test_dataset:
        ret.append(
            [
                head_rank_result_map[tuple(triple)],
                tail_rank_result_map[tuple(triple)]
            ]
        )

    ret = np.array(ret)
    mr_score = np.mean(ret) 
    mrr_score = np.mean(1/ret)
    hits_1 = np.mean(ret <= 1)
    hits_3 = np.mean(ret <= 3)
    hits_10 = np.mean(ret <= 10)
    return {
        "results": ret,
        "mr": mr_score,
        "mrr": mrr_score,
        "hits_1": hits_1,
        "hits_3": hits_3,
        "hits_10": hits_10
    }