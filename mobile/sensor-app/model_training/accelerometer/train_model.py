"""CLI orchestrator for the accelerometer pipeline. Mirrors the camera pipeline's style."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end accelerometer drowsiness pipeline.",
    )
    parser.add_argument(
        "step",
        choices=["train", "export", "all"],
        default="all",
        nargs="?",
        help="Pipeline step to run.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip leave-one-driver-out CV (only used by 'train'/'all').")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic-range quantization during export.")
    args = parser.parse_args()

    if args.step in {"train", "all"}:
        from load_dataset import load_sessions, summarize
        from train import leave_one_driver_out, train_final

        sessions = load_sessions()
        summarize(sessions)
        if not args.skip_cv:
            cv = leave_one_driver_out(
                sessions, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            )
            print("\nCV summary:", cv["summary"])
        train_final(
            sessions, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        )

    if args.step in {"export", "all"}:
        from export_tflite import export_tflite
        export_tflite(quantize=args.quantize)


if __name__ == "__main__":
    main()
