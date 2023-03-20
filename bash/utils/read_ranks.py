import os
import csv
import numpy as np

# Read ranks.csv from {csv_path}
def read_ranks(csv_path):
    return np.loadtxt(csv_path, delimiter=",", dtype=int)
