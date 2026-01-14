#!/usr/bin/env python
"""
Master training script - runs complete pipeline.

IMPORTANT:
This script is NOT evaluated by the competition.
It exists only to:
1. Train models
2. Export ONNX
3. Sanity-check correlation locally

Evaluation ONLY uses:
- solution.py
- predictor_final.onnx
"""
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Complete training pipeline")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing train.parquet and valid.parquet")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on: cpu or cuda")
    parser.add_argument("--encoder-epochs", type=int, default=50)
    parser.add_argument("--predictor-epochs-p1", type=int, default=80)
    parser.add_argument("--predictor-epochs-p2", type=int, default=40)
    parser.add_argument("--batch-size-encoder", type=int, default=32)
    parser.add_argument("--batch-size-predictor", type=int, default=8)
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-predictor", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")

    args = parser.parse_args()

    if args.batch_size_predictor > 8:
        print("⚠ WARNING: batch_size_predictor > 8 can hurt correlation stability")

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.parquet"
    valid_path = data_dir / "valid.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    print("=" * 80)
    print("WUNDER LOB PREDICTORIUM — TRAINING PIPELINE")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Train data: {train_path}")
    print(f"Valid data: {valid_path if valid_path.exists() else 'NOT FOUND'}")
    print("=" * 80)

    start_time = time.time()

    # ===================== PHASE 1 =====================
    if not args.skip_encoder:
        print("\nPHASE 1 — SELF-SUPERVISED ENCODER")
        from train_encoder import train_encoder

        train_encoder(
            data_path=str(train_path),
            epochs=args.encoder_epochs,
            batch_size=args.batch_size_encoder,
            lr=1e-3,
            device=args.device,
            save_path="models/encoder_pretrained.pt"
        )
    else:
        print("\n⏭ Skipping encoder pretraining")

    # ===================== PHASE 2 & 3 =====================
    if not args.skip_predictor:
        print("\nPHASE 2 & 3 — SUPERVISED PREDICTOR")
        from train_predictor import train_predictor

        train_predictor(
            data_path=str(train_path),
            encoder_path="models/encoder_pretrained.pt",
            epochs_phase1=args.predictor_epochs_p1,
            epochs_phase2=args.predictor_epochs_p2,
            batch_size=args.batch_size_predictor,
            lr_phase1=5e-4,
            lr_phase2=5e-5,
            device=args.device,
            save_path="models/predictor_final.pt"
        )
    else:
        print("\n⏭ Skipping predictor training")

    # ===================== ONNX EXPORT =====================
    if not args.skip_export:
        print("\nONNX EXPORT")
        from export_onnx import export_to_onnx

        export_to_onnx(
            encoder_path="models/encoder_pretrained.pt",
            predictor_path="models/predictor_final.pt",
            output_path="models/predictor_final.onnx",
            device=args.device
        )
    else:
        print("\n⏭ Skipping ONNX export")

    # ===================== VALIDATION =====================
    if not args.skip_validation and valid_path.exists():
        print("\nVALIDATION (Metric Sanity Check Only)")
        try:
            from validate import evaluate_model
            from solution import PredictionModel

            metrics = evaluate_model(
                PredictionModel,
                str(valid_path)
            )

            print("\nValidation Results:")
            print(f"  Corr t0:  {metrics['corr_t0']:.4f}")
            print(f"  Corr t1:  {metrics['corr_t1']:.4f}")
            print(f"  Corr avg: {metrics['corr_avg']:.4f}")

        except Exception as e:
            print(f"\n⚠ Validation failed: {e}")
    else:
        print("\n⏭ Validation skipped")

    # ===================== SUMMARY =====================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print("\nArtifacts produced:")
    print("  - models/encoder_pretrained.pt")
    print("  - models/predictor_final.pt")
    print("  - models/predictor_final.onnx")
    print("\nSUBMISSION:")
    print("  zip -r submission.zip solution.py predictor_final.onnx")
    print("=" * 80)


if __name__ == "__main__":
    main()
