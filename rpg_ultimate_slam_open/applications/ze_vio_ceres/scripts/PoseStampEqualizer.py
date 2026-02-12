#!/usr/bin/env python3
import numpy as np

# Load groundtruth and trajectory files
gt_path = "/media/sebnem/T7/realtime_tests/manual_long_17_04_25/groundtruth.txt"
traj_path = "/media/sebnem/T7/realtime_tests/manual_long_17_04_25/trajectory.txt"

gt_data = np.loadtxt(gt_path)
traj_data = np.loadtxt(traj_path)

# Match timestamps by finding closest match within max_diff
max_diff = 0.02  # seconds
matched_gt, matched_traj = [], []

for gt_row in gt_data:
    gt_time = gt_row[0]
    diffs = np.abs(traj_data[:, 0] - gt_time)
    min_idx = np.argmin(diffs)

    if diffs[min_idx] < max_diff:
        matched_gt.append(gt_row)
        matched_traj.append(traj_data[min_idx])

# Convert to arrays
gt_aligned = np.array(matched_gt)
traj_aligned = np.array(matched_traj)

# Save aligned data
np.savetxt("/media/sebnem/T7/realtime_tests/manual_long_17_04_25/aligned_gt.txt", gt_aligned, fmt="%.9f")
np.savetxt("/media/sebnem/T7/realtime_tests/manual_long_17_04_25/aligned_traj.txt", traj_aligned, fmt="%.9f")

print(f"Eşleşen poz sayısı: {len(gt_aligned)}")
