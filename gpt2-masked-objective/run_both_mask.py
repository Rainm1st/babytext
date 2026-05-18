import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    r = subprocess.run([str(x) for x in cmd], cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only full pipeline: preflight + train + eval + fill tables.")
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-max-samples", type=int, default=-1)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="babylm-strict")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="weichunzhou527-xi-an-jiaotong-liverpool-university",
    )
    parser.add_argument("--resume-mask5", type=str, default=None)
    parser.add_argument("--resume-mask10", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    exp = root / "experiments" / "gpt2-masked-objective"
    py = sys.executable

    preflight = exp / "prepare_experiment.py"
    run_train = exp / "run_train.py"
    eval_script = exp / "evaluate_and_fill_tables.py"
    out_dir = Path(os.environ.get("GPT2_MASKED_RUN_BASE", "/data0/language/babylm_runs/gpt2_masked_objective"))
    result_dir = exp / "results"

    run_cmd([py, preflight, "--no-baseline-compare"], root)
    train_mask5_cmd = [
        py,
        run_train,
        "--model",
        "mask5",
        "--project-root",
        root,
        "--num-workers",
        args.num_workers,
        "--logging-steps",
        args.logging_steps,
    ]
    train_mask10_cmd = [
        py,
        run_train,
        "--model",
        "mask10",
        "--project-root",
        root,
        "--num-workers",
        args.num_workers,
        "--logging-steps",
        args.logging_steps,
    ]
    if args.resume_mask5:
        train_mask5_cmd.extend(["--resume-checkpoint", args.resume_mask5])
    if args.resume_mask10:
        train_mask10_cmd.extend(["--resume-checkpoint", args.resume_mask10])
    if args.wandb:
        train_mask5_cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-entity",
                args.wandb_entity,
            ]
        )
        train_mask10_cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-entity",
                args.wandb_entity,
            ]
        )

    run_cmd(train_mask5_cmd, root)
    run_cmd(train_mask10_cmd, root)

    mask5_summary = out_dir / "gpt2_mask5_run1" / "run_summary.json"
    mask10_summary = out_dir / "gpt2_mask10_run1" / "run_summary.json"
    if not mask5_summary.exists() or not mask10_summary.exists():
        raise FileNotFoundError("Missing run_summary.json after training.")

    m5 = json.loads(mask5_summary.read_text(encoding="utf-8"))
    m10 = json.loads(mask10_summary.read_text(encoding="utf-8"))
    compare = {
        "baseline_reference": "official_gpt2_baseline",
        "run_mask5": m5,
        "run_mask10": m10,
        "deltas": {
            "masked_alpha": float(m10["masked_alpha"]) - float(m5["masked_alpha"]),
            "words_seen": float(m10["words_seen"]) - float(m5["words_seen"]),
            "global_step": float(m10["global_step"]) - float(m5["global_step"]),
        },
    }
    compare_path = result_dir / "mask_runs_summary_compare.json"
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", compare_path)

    run_cmd(
        [
            py,
            eval_script,
            "--mask5-summary",
            mask5_summary,
            "--mask10-summary",
            mask10_summary,
            "--raw-csv",
            result_dir / "raw_scores_template.csv",
            "--delta-csv",
            result_dir / "delta_vs_baseline_template.csv",
            "--output-json",
            result_dir / "eval_mask_runs.json",
            "--max-samples",
            args.eval_max_samples,
        ],
        root,
    )

    print("Done: train + evaluate + fill tables")


if __name__ == "__main__":
    main()
