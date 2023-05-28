import torch
from torch import FloatTensor

class Loss(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def __call__(
        self,
        pos_ret: FloatTensor,
        neg_ret: FloatTensor,
    ):
        return self.forward(pos_ret, neg_ret)

    def forward(
        self,
        pos_ret: FloatTensor, 
        neg_ret: FloatTensor
    ):
        raise NotImplementedError

class MulticlassNLL(Loss):
    def forward(
        self, 
        pos_ret: FloatTensor, # (batch_size,)
        neg_ret: FloatTensor # (batch_size, eta)
    ):
        pos_exp = torch.exp(pos_ret)
        neg_exp = torch.exp(neg_ret)
        softmax_score = pos_exp / (torch.mean(neg_exp, dim=-1) + pos_exp)
        loss = torch.sum(-torch.log(softmax_score))
        return loss

class NLL(Loss):
    def forward(
        self,
        pos_ret: FloatTensor, # (batch_size,)
        neg_ret: FloatTensor # (batch_size, eta)
    ):
        pos_ret = pos_ret.unsqueeze(dim=1).tile([1, neg_ret.shape[1]]).view(-1).contiguous()
        neg_ret = neg_ret.view(-1).contiguous()
        exp = torch.exp(
            torch.cat((-pos_ret, neg_ret), dim=0)
        )
        loss = torch.mean(torch.log(1.0 + exp))
        return loss

        
LOSS = {
    "multiclass_nll": MulticlassNLL,
    "nll": NLL
}