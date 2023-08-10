import os
import torch
from torch.optim import AdamW
from pykeen.models import ERModel
from torch.utils.tensorboard import SummaryWriter
import tqdm

from dataset import KGDataloader
from .loss import LOSS
from .eval import eval, RankingStrategy

def _head_forward(model, batch, loss_func, device):
    data = batch["positives"].to(device)
    ret = model.score_h(data[:, 1:])
    loss = loss_func.target_foward(ret, data[:, 0])
    return loss

def _tail_forward(model, batch, loss_func, device):
    data = batch["positives"].to(device)
    ret = model.score_t(data[:, :2])
    loss = loss_func.target_forward(ret, data[:, 2])
    return loss

def _normal_forward(model, batch, loss_func, device):
    pos = batch["positives"].to(device)
    neg = batch["negatives"].to(device)

    pos_ret = model(pos[:, 0], pos[:, 1], pos[:, 2], mode=None).view(-1).contiguous()

    _neg = neg.view(-1, 3).contiguous()
    neg_ret = model(
        _neg[:, 0], _neg[:, 1], _neg[:, 2], mode=None
    ).view(neg.shape[:2]).contiguous()

    loss = loss_func(pos_ret, neg_ret) + model.collect_regularization_term()

    return loss

FORWARD = {
    "head": _head_forward, 
    "tail": _tail_forward,
    "normal": _normal_forward
}


def train(
    model: ERModel,
    config,
    train_dataset,
    save_dir: str,
    valid_dataset = None
):
    epochs = config.train.epochs
    lr = config.train.lr
    eta = config.train.eta
    
    batch_size = config.train.get("batch_size", None)
    batch_count = config.train.get("batch_count", None)
    dataloader = KGDataloader(train_dataset, eta, batch_size, batch_count, mode = config.train.get("mode", "ht"))
    
    loss_func = LOSS[config.train.loss](**config.train.get("loss_kwargs", {}))
    print = tqdm.tqdm.write

    forward_type = config.train.get("forward_type", "normal")
    print(f"Forward Type: {forward_type}")
    _forward = FORWARD[forward_type]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train on {device}")

    model.to(device)
    model.train()

    eval_step = config.train.get("eval_step", None)
    save_step = config.train.get("save_step", None)

    optimizer = AdamW(model.parameters(), lr, **config.train.get("optimizer_kwargs", {}))
    
    step = 0
    total_step = epochs * len(dataloader)

    tb_log_path = os.path.join(save_dir, "tb_log")
    os.makedirs(tb_log_path, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_path)

    start_step = config.train.get("start_step", 0)

    for epoch in range(epochs):
        for batch in tqdm.tqdm(dataloader, desc="Train"):
            step += 1
            if step < start_step:
                continue

            loss = _forward(model, batch, loss_func, device)
            loss.backward()
            print(f"Epoch: {epoch} Step: {step}/{total_step} Loss: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()
            model.post_parameter_update()

            tb_writer.add_scalar("loss", loss.item(), step)

            if save_step is not None and step % save_step == 0:
                print("Save Model")
                sd = model.state_dict()
                sd_path = os.path.join(save_dir, f"epoch_{epoch}_step_{step}.bin")
                torch.save(sd, sd_path)
                

            if eval_step is not None and step % eval_step == 0:
                print("Evaluation")
                eval_ret = eval(
                    model, 
                    train_dataset,
                    valid_dataset,
                    batch_size = config.eval.batch_size,
                    filter_datasets = (train_dataset, valid_dataset),
                    ranking_strategy = RankingStrategy[config.eval.get("ranking_strategy", "worst").upper()],
                    mode = config.eval.get("mode", "ht").lower()
                )
                mrr = eval_ret["mrr"]
                mr = eval_ret["mr"]
                hits_1 = eval_ret["hits_1"]
                hits_3 = eval_ret["hits_3"]
                hits_10 = eval_ret["hits_10"]
                print(f"Eval Result | mrr: {mrr:.2f} | mr: {mr:.2f} | hits_1: {hits_1: .2f} | hits_3: {hits_3:.2f} | hits_10: {hits_10:.2f}")
                model.train()
                tb_writer.add_scalar("mr", mr, step)
                tb_writer.add_scalar("mrr", mrr, step)
                tb_writer.add_scalar("hits_1", hits_1, step)
                tb_writer.add_scalar("hits_3", hits_3, step)
                tb_writer.add_scalar("hits_10", hits_10, step)


    sd = model.state_dict()               
    sd_path = os.path.join(save_dir, "model.bin")
    torch.save(sd, sd_path)
    print(f"Model Weights Saved at {sd_path}")
    return model