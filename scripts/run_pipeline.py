"""
run_pipeline.py
---------------
IE 423 — Term Project
End-to-end pipeline runner.

Usage (from project root):
    python scripts/run_pipeline.py              # full pipeline (stages 01-08)
    python scripts/run_pipeline.py --from 04    # skip preprocessing/EDA
    python scripts/run_pipeline.py --stage 06   # run one stage only

Pipeline stages:
    01  load_data                  preprocessing
    02  preprocess_data            preprocessing
    03  basic_eda                  visualization
    04  build_tfidf                feature_engineering
    05  build_style_features       feature_engineering (RQ3 placeholder)
    06  train_models               modeling
    07  evaluate_models            evaluation
    08  interpretability_analysis  visualization
"""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

STAGES = {
    "01": os.path.join("scripts", "preprocessing", "01_load_data.py"),
    "02": os.path.join("scripts", "preprocessing", "02_preprocess_data.py"),
    "03": os.path.join("scripts", "visualization", "03_basic_eda.py"),
    "04": os.path.join("scripts", "feature_engineering", "04_build_tfidf.py"),
    "05": os.path.join("scripts", "feature_engineering", "05_build_style_features.py"),
    "06": os.path.join("scripts", "modeling", "06_train_models.py"),
    "07": os.path.join("scripts", "evaluation", "07_evaluate_models.py"),
    "08": os.path.join("scripts", "visualization", "08_interpretability_analysis.py"),
}


def run_stage(stage_id: str) -> None:
    """Run a single pipeline stage by ID."""
    script = STAGES[stage_id]
    script_path = os.path.join(PROJECT_ROOT, script)

    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Stage {stage_id}: {script}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Stage {stage_id} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[OK] Stage {stage_id} completed.")


def main():
    parser = argparse.ArgumentParser(description="Run the IE423 ML pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from",
        dest="from_stage",
        choices=list(STAGES.keys()),
        help="Start pipeline from this stage (inclusive)",
    )
    group.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        help="Run only this stage",
    )
    args = parser.parse_args()

    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        stage_ids = list(STAGES.keys())
        start_idx = stage_ids.index(args.from_stage)
        stages_to_run = stage_ids[start_idx:]
    else:
        stages_to_run = list(STAGES.keys())

    print("IE423 MBTI Personality Prediction Pipeline")
    print(f"Running stages: {', '.join(stages_to_run)}")

    for stage_id in stages_to_run:
        run_stage(stage_id)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
