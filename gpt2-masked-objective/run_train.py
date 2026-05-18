import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only launcher for one mask run.")
    parser.add_argument("--model", choices=["mask5", "mask10", "mask20", "mask50"], required=True)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", default=str(default_root))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="babylm-strict")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="weichunzhou527-xi-an-jiaotong-liverpool-university",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    exp = root / "experiments" / "gpt2-masked-objective"
    train_script = exp / "train_hybrid_gpt2.py"

    run_map = {
        "mask5": ("run_mask5.json", "gpt2_mask5_run1"),
        "mask10": ("run_mask10.json", "gpt2_mask10_run1"),
        "mask20": ("run_mask20.json", "gpt2_mask20_run1"),
        "mask50": ("run_mask50.json", "gpt2_mask50_run1"),
    }
    cfg_name, out_name = run_map[args.model]
    cfg = exp / "configs" / cfg_name
    out_base = Path(os.environ.get("GPT2_MASKED_RUN_BASE", "/data0/language/babylm_runs/gpt2_masked_objective"))
    out = out_base / out_name

    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(cfg),
        "--output-dir",
        str(out),
        "--num-workers",
        str(args.num_workers),
        "--logging-steps",
        str(args.logging_steps),
    ]
    if args.resume_checkpoint:
        cmd.extend(["--resume-checkpoint", str(args.resume_checkpoint)])
    if args.wandb:
        cmd.extend(
            [
                "--wandb",
                "--wandb-project",
                str(args.wandb_project),
                "--wandb-entity",
                str(args.wandb_entity),
            ]
        )

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
