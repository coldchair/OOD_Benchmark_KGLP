import os
from config import MODELS_ROOT_PATH
from config import IMAGE_ROOT_PATH
from utils.dir import get_models_list
from utils.read_ranks import read_ranks
from datasets import Dataset
from degree.s_plus_o_100_buckets import draw

models_list = get_models_list(MODELS_ROOT_PATH)

for model in models_list:
    dataset = Dataset(MODELS_ROOT_PATH, model)
    ranks = read_ranks(os.path.join(MODELS_ROOT_PATH, model, "ranks.csv"))
    draw(dataset, ranks, os.path.join(IMAGE_ROOT_PATH, model))