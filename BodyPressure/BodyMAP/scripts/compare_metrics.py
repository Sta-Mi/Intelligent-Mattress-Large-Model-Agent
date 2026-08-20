import argparse
import json
from pathlib import Path


SUMMARY_KEYS = ("MPJPE", "PVE", "height", "chest", "waist", "hips", "v2vP", "v2vP_1EA", "v2vP_2EA")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BodyMAP metrics JSON files.")
    parser.add_argument(
        "runs",
        nargs="+",
        metavar="LABEL=METRICS_JSON",
        help="Example: both=/path/metrics.json depth=/path/metrics.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_runs(specs):
    runs = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, got {spec}")
        label, path = spec.split("=", 1)
        runs[label] = json.loads(Path(path).read_text())
    return runs


def summarize(runs):
    output = {"runs": {}, "comparisons": {}}
    for label, metrics in runs.items():
        output["runs"][label] = {cover: {key: values[key] for key in SUMMARY_KEYS} for cover, values in metrics.items()}
        uncover = metrics["uncover"]
        covered = {
            key: (metrics["cover1"][key] + metrics["cover2"][key]) / 2
            for key in SUMMARY_KEYS
        }
        output["runs"][label]["covered_mean"] = covered
        output["runs"][label]["covered_vs_uncover_percent"] = {
            key: 100.0 * (covered[key] - uncover[key]) / uncover[key]
            if uncover[key] != 0
            else None
            for key in SUMMARY_KEYS
        }

    labels = list(runs)
    if len(labels) > 1:
        baseline = labels[0]
        for label in labels[1:]:
            output["comparisons"][f"{label}_minus_{baseline}_overall"] = {
                key: runs[label]["overall"][key] - runs[baseline]["overall"][key]
                for key in SUMMARY_KEYS
            }
            output["comparisons"][f"{label}_minus_{baseline}_percent_overall"] = {
                key: 100.0
                * (runs[label]["overall"][key] - runs[baseline]["overall"][key])
                / runs[baseline]["overall"][key]
                if runs[baseline]["overall"][key] != 0
                else None
                for key in SUMMARY_KEYS
            }
    return output


def main():
    args = parse_args()
    result = summarize(load_runs(args.runs))
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"Comparison saved to {args.out}")
    print(rendered)


if __name__ == "__main__":
    main()
