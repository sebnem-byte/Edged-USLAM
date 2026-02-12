#!/usr/bin/env python3
import numpy as np

def load_tum_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                data.append([float(x) for x in parts])
    return np.array(data)

def save_tum_file(path, data):
    with open(path, 'w') as f:
        for row in data:
            f.write(" ".join(f"{x:.9f}" for x in row) + "\n")

# load files
gt = load_tum_file("/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_gt.txt")
aligned = load_tum_file("/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_traj.txt")

# compute initial position difference (translation only)
delta = gt[0, 1:4] - aligned[0, 1:4]

# apply translation shift to aligned traj
aligned[:, 1:4] += delta

# save corrected result
save_tum_file("traj_aligned_shifted.txt", aligned)
