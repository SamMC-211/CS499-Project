import argparse

# Args: step, epochs, quantize
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end drowsiness pipeline: preprocess, train, and export."
    )
    parser.add_argument(
        "step",
        choices=["preprocess", "train", "export", "all"],
        default="all",
        nargs="?",
        help="Pipeline step to run.",
    )
    # how many training epochs will be ran
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply post-training dynamic range quantization during export.",
    )
    args = parser.parse_args()

    if args.step in {"preprocess", "all"}:
        from preprocess import preprocess_dataset

        total = preprocess_dataset()
        print(f"Preprocessing complete. Wrote {total} images.")

    if args.step in {"train", "all"}:
        from train import train_and_save

        train_and_save(epochs=args.epochs)

    if args.step in {"export", "all"}:
        from export_tflite import export_tflite

        export_tflite(quantize=args.quantize)


if __name__ == "__main__":
    main()
