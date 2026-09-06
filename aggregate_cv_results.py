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

    # experiment_name matching is prefix-based, so a name like 'eegpt_adhd_fold0_pretrained'
    # also starts with 'eegpt_adhd_fold0' -- use --exclude/--suffix to disambiguate
    # variants that share a common prefix stem:
    python aggregate_cv_results.py --prefix eegpt_adhd_fold --suffix _pretrained   # pretrained only
    python aggregate_cv_results.py --prefix eegpt_adhd_fold --exclude _pretrained  # scratch only
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


def find_metric_records(run_root: str, prefix: str, suffix: str = '', exclude: str = '') -> list:
    pattern = os.path.join(run_root, 'log', 'baseline', '*', '*', 'best_test_metrics_*.json')
    matches = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        name = data.get('experiment_name', '')
        if not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        if exclude and exclude in name:
            continue
        data['_source_path'] = path
        matches.append(data)
    return matches


def aggregate(records: list) -> dict:
    """Per-metric NaN-aware aggregation.

    A metric can be legitimately undefined for a specific run -- e.g. AUROC
    has no meaning when a LOSO fold's held-out subject has only one class in
    their trials, so sklearn has nothing to rank against. Rather than let one
    NaN poison the whole-cohort mean (plain sum()/n does exactly that), each
    metric is aggregated only over the runs where it actually has a value.
    Runs excluded from one metric this way still fully count for every other
    metric -- nothing is dropped at the run level, only per-metric.
    """
    per_metric = defaultdict(list)
    n_total = len(records)
    for rec in records:
        for k, v in rec.get('metrics', {}).items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                per_metric[k].append(v)

    results = {}
    for metric, values in sorted(per_metric.items()):
        n = len(values)
        n_dropped = n_total - n
        if n == 0:
            mean = std = ci_half_width = float('nan')
        else:
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
            'n_dropped': n_dropped,
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
    parser.add_argument('--suffix', default='',
                         help="optional: only keep names ending with this, e.g. '_pretrained'")
    parser.add_argument('--exclude', default='',
                         help="optional: skip names containing this substring, e.g. '_pretrained' "
                              "to get scratch-only results when pretrained runs share the same prefix")
    parser.add_argument('--run_root', default='assets/run')
    args = parser.parse_args()

    records = find_metric_records(args.run_root, args.prefix, args.suffix, args.exclude)
    if not records:
        print(f"No best_test_metrics_*.json files found with experiment_name starting "
              f"with '{args.prefix}'"
              f"{f' and ending with {args.suffix!r}' if args.suffix else ''}"
              f"{f' and excluding {args.exclude!r}' if args.exclude else ''} "
              f"under {args.run_root}/log/baseline/")
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
    print(f"\n{'metric':32s} {'n':>4s} {'mean':>9s} {'std':>9s} {'95% CI':>22s}")
    for metric, s in results.items():
        n_str = f"{s['n']}*" if s['n_dropped'] else f"{s['n']}"
        print(f"{metric:32s} {n_str:>4s} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"[{s['ci95_low']:>9.4f}, {s['ci95_high']:>9.4f}]")

    if any(s['n_dropped'] for s in results.values()):
        print("\n* n < total matched runs: those runs had no value (NaN) for this specific "
              "metric -- e.g. AUROC is undefined when a LOSO fold's held-out subject has only "
              "one class in their trials -- and were excluded from just this metric's "
              "aggregate. They're still fully included in every other metric above.")


if __name__ == '__main__':
    main()
