import os
from omegaconf import OmegaConf
import numpy as np
import torch
from pykeen.models import (
    ConvE,
    DistMult,
    TransE,
)

from dataset import KGDataset
from utils import train, eval, RankingStrategy

MODEL = {
    "TransE": TransE,
    "ConvE": ConvE,
    "DistMult": DistMult
}

def load_data(dir_path: str):
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    train_file_path = os.path.join(dir_path, "train.tsv")
    train_dataset = KGDataset.load_csv(train_file_path)

    valid_file_path = os.path.join(dir_path, "train.tsv")
    if os.path.exists(valid_file_path):
        valid_dataset = KGDataset.load_csv(
            valid_file_path,
            entity_to_id=train_dataset.entity_to_id,
            relation_to_id=train_dataset.relation_to_id
        )

    test_file_path = os.path.join(dir_path, "test.tsv")
    test_dataset = KGDataset.load_csv(
        test_file_path,
        entity_to_id=train_dataset.entity_to_id,
        relation_to_id=train_dataset.relation_to_id
    )
    return {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }

def main(config):
    datasets = load_data(config.dataset_dir_path)
    model_name = config.model.pop("name")
    model = MODEL[model_name](
        triples_factory = datasets["train"].to_triples_factory(),
        **config.model
    )
    if config.model_ckpt_path and os.path.isfile(config.model_ckpt_path):
        sd = torch.load(config.model_ckpt_path)
        model.load_state_dict(sd)
        print(f"Load Model Weights From {config.model_ckpt_path}")

    if config.do_train:
        model = train(
            model,
            config,
            datasets["train"],
            config.output_dir_path,
            datasets["valid"]
        )

    ret = eval(
        model,
        datasets["train"],
        datasets["test"],
        batch_size=config.eval.batch_size,
        ranking_strategy=RankingStrategy[config.eval.get("ranking_strategy", "worst").upper()],
        filter_datasets=tuple(datasets.values())
    )
    print("Result | mr: {:.2f} | mrr: {:.2f} | hits_1: {:.2f} | hits_3: {:.2f} | hits_10: {:.2f}".format(
        ret["mr"],
        ret["mrr"],
        ret["hits_1"],
        ret["hits_3"],
        ret["hits_10"]
    ))
    result_path = os.path.join(config.output_dir_path, "results.tsv")
    np.savetxt(result_path, ret["results"], delimiter="\t", fmt="%d")
    print(f"Results Saved in {result_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Model Config File Path", required=True)
    parser.add_argument("--dataset_dir_path", type=str, help="Dataset Directory Path", required=True)
    parser.add_argument("--model_ckpt_path", type=str, help="Model Checkpoint Path", default=None)
    parser.add_argument("--train", default=False, help="Train Model", action="store_true")
    parser.add_argument("--output_dir_path", type=str, help="Results Save Directory Path", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir_path, exist_ok=True)
    config = OmegaConf.load(args.config_path)
    config.do_train = args.train
    config.dataset_dir_path = args.dataset_dir_path
    config.model_ckpt_path = args.model_ckpt_path
    config.output_dir_path = args.output_dir_path
    main(config)
