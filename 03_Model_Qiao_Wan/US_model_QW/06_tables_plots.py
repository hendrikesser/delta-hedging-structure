import numpy as np


def print_summary_table(overall_gains, bucket_gains, buckets):
    """Print mean and std of gain ratios across NUM_RUNS runs.

    Args:
        overall_gains:  list of overall gain ratios (one per run)
        bucket_gains:   dict {bucket_value: [gain_run1, ..., gain_runN]}
        buckets:        ordered list of delta-bucket centre values
    """
    print("=" * 58)
    print(f"{'Delta Bucket':>12} | {'Mean Gain Ratio':>17} | {'Std Dev':>18}")
    print("=" * 58)

    for b in buckets:
        gains = bucket_gains.get(b, [])
        if gains:
            print(f"{b:>12.1f} | {np.mean(gains):>17.4f} | {np.std(gains, ddof=1):>17.5f}")

    print("-" * 58)
    print(f"{'OVERALL':>12} | {np.mean(overall_gains):>17.4f} | {np.std(overall_gains, ddof=1):>17.5f}")
    print("=" * 58)
