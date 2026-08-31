"""
aggregate_cv_results.py

Aggregates best_test_metrics_*.json files (saved by AbstractTrainer at the end of
each run, from the best-val-AUROC epoch) across multiple runs of the same k-fold
config, computing mean +/- 95% confidence interval per metric. Uses a t-distribution
critical value (not a normal approximation), which is the correct choice for the
small n=5 fold count used in this benchmark's k-fold CV.

Run directories are timestamp-named, not experiment-named, so this matches runs by
the `experiment_name` field saved inside each JSON file rather than by path.

Usage:
    python aggregate_cv_results.py --prefix neurogpt_adhd_fold
    python aggregate_cv_results.py --prefix neurogpt_crown_scratch_fold --run_root assets/run
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Two-sided 95% CI t-table fallback (df=1..10) if scipy isn't installed.
_T_TABLE_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_critical(n: int, confidence: float = 0.95) -> float:
    df = max(1, n - 1)
    if HAVE_SCIPY:
        return float(stats.t.ppf(1 - (1 - confidence) / 2, df))
    return _T_TABLE_95.get(df, 1.96)  # normal approx for larger df than the table covers


def find_metric_records(run_root: str, prefix: str) -> list:
    pattern = os.path.join(run_root, 'log', 'baseline', '*', '*', 'best_test_metrics_*.json')
    matches = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get('experiment_name', '').startswith(prefix):
            data['_source_path'] = path
            matches.append(data)
    return matches


def aggregate(records: list) -> dict:
    per_metric = defaultdict(list)
    for rec in records:
        for k, v in rec.get('metrics', {}).items():
            if isinstance(v, (int, float)):
                per_metric[k].append(v)

    results = {}
    for metric, values in sorted(per_metric.items()):
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
            sem = std / math.sqrt(n)
            ci_half_width = t_critical(n) * sem
        else:
            std = float('nan')
            ci_half_width = float('nan')
        results[metric] = {
            'n': n,
            'mean': mean,
            'std': std,
            'ci95_low': mean - ci_half_width,
            'ci95_high': mean + ci_half_width,
            'values': values,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prefix', required=True,
                         help="experiment_name prefix to match, e.g. 'neurogpt_adhd_fold'")
    parser.add_argument('--run_root', default='assets/run')
    args = parser.parse_args()

    records = find_metric_records(args.run_root, args.prefix)
    if not records:
        print(f"No best_test_metrics_*.json files found with experiment_name starting "
              f"with '{args.prefix}' under {args.run_root}/log/baseline/")
        return

    print(f"Found {len(records)} run(s) matching prefix '{args.prefix}':")
    for rec in sorted(records, key=lambda r: r.get('experiment_name', '')):
        print(f"  {rec.get('experiment_name')}  "
              f"(dataset_config={rec.get('dataset_config')}, seed={rec.get('seed')}, "
              f"best_epoch={rec.get('best_epoch')})")

    if not HAVE_SCIPY:
        print("\n(scipy not installed -- using a t-table fallback limited to df<=10; "
              "install scipy for exact critical values at any fold count.)")

    results = aggregate(records)
    print(f"\n{'metric':32s} {'n':>3s} {'mean':>9s} {'std':>9s} {'95% CI':>22s}")
    for metric, s in results.items():
        print(f"{metric:32s} {s['n']:>3d} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"[{s['ci95_low']:>9.4f}, {s['ci95_high']:>9.4f}]")


if __name__ == '__main__':
    main()
