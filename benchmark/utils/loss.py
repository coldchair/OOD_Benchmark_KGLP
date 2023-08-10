import torch
from torch import FloatTensor, IntTensor

class Loss(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._reduction_method = kwargs.get("reduction_method", "sum")
    
    def __call__(
        self,
        pos_ret: FloatTensor,
        neg_ret: FloatTensor,
    ):
        return self.forward(pos_ret, neg_ret)

    def _reduce_sapmle_loss(self, loss):
        assert loss.ndim > 1
        if self._reduction_method == "mean":
            return torch.mean(loss, dim=-1)
        else:
            return torch.sum(loss, dim=-1)

    def forward(
        self,
        pos_ret: FloatTensor, 
        neg_ret: FloatTensor
    ):
        raise NotImplementedError

    def target_forward(
        self,
        ret: FloatTensor,
        target: IntTensor,
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
        softmax_score = pos_exp / (self._reduce_sapmle_loss(neg_exp) + pos_exp)
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

        
class BCEWithLogits(Loss):
    def forward(
        self,
        pos_ret: FloatTensor, # (batch_size,)
        neg_ret: FloatTensor, # (batch_size, eta)
    ):
        pos_p = torch.nn.functional.sigmoid(pos_ret).view(-1)
        neg_p = torch.nn.functional.sigmoid(neg_ret).view(-1)
        n = pos_p.size(0) + neg_p.size(0)

        loss = - (torch.log(pos_p).sum() + torch.log(1-neg_p).sum()) / n

        return loss

    def target_forward(
        self,
        ret: FloatTensor, # (batch_size, num_ent)
        target: IntTensor, # (batch_size,)
    ):
        ret = torch.nn.functional.sigmoid(ret) 
        n = ret.view(-1).size(0)

        target = target.long()
        log_ret = torch.log(ret)
        pos_sum = log_ret[torch.arange(ret.size(0), device=target.device, dtype=target.dtype), target].sum() 
        log_ret_1 = torch.log(1-ret)
        pos_sum_1 = log_ret_1[torch.arange(ret.size(0), device=target.device, dtype=target.dtype), target].sum() 
        total_sum_1 = log_ret_1.sum()
        neg_sum_1 = total_sum_1 - pos_sum_1

        loss = - (pos_sum + neg_sum_1) / n

        return loss

class NSSA(Loss):
    def __init__(self, margin, adversarial_temperature, reduction="mean"):
        from pykeen.losses import NSSALoss
        self.loss_func = NSSALoss(margin, adversarial_temperature, reduction)

    def forward(
        self, 
        pos_ret: FloatTensor, 
        neg_ret: FloatTensor, 
    ):

        return self.loss_func.process_slcwa_scores(pos_ret, neg_ret)
        


LOSS = {
    "multiclass_nll": MulticlassNLL,
    "nll": NLL,
    "BCEWithLogits": BCEWithLogits,
    "NSSA": NSSA
}