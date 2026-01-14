"""
Self-supervised encoder pretraining script.
CRITICAL: Progressive masking, no temporal leakage, gradient stability
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from src.dataset import SequenceDataset
from src.encoder import TemporalEncoder, MaskedReconstructionTask, ProgressiveMasking


def train_encoder(
    data_path='data/train.parquet',
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device='cpu',
    save_path='models/encoder_pretrained.pt'
):
    device = torch.device(device)
    print(f"Training on {device}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = SequenceDataset(data_path, store_numpy=True)

    # 🔧 IMPORTANT: shuffle sequences ONCE, not per epoch
    np.random.shuffle(dataset.sequences)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # DO NOT SHUFFLE TEMPORALLY
        num_workers=4,
        pin_memory=device.type == 'cuda'
    )

    encoder = TemporalEncoder(d_in=64, d_hidden=64, d_gru=96).to(device)
    task = MaskedReconstructionTask(encoder, d_in=64).to(device)

    optimizer = torch.optim.AdamW(
        task.parameters(), lr=lr, weight_decay=1e-5
    )

    masking_schedule = ProgressiveMasking(
        start_prob=0.15,
        end_prob=0.40,
        total_epochs=epochs
    )

    print("\nStarting pretraining...")
    for epoch in range(epochs):
        task.train()
        epoch_losses = []

        mask_prob = masking_schedule.get_mask_prob(epoch)
        span = int(5 + epoch / epochs * 5)  # 5 → 10

        # 🔧 Freeze stem after early geometry learning
        if epoch == 10:
            for p in encoder.stem.parameters():
                p.requires_grad = False

        # 🔧 LR decay after structure learned
        if epoch == int(epochs * 0.6):
            for g in optimizer.param_groups:
                g['lr'] *= 0.3

        for batch in loader:
            x_full = batch['x'].to(device)
            x = x_full[:, 99:, :]  # exclude warm-up

            optimizer.zero_grad()
            loss = task(x, mask_prob=mask_prob, mask_span_length=span)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        std_loss = np.std(epoch_losses)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Mask {mask_prob:.3f} | "
            f"Span {span:02d} | "
            f"Loss {mean_loss:.4f} ± {std_loss:.4f}"
        )

    torch.save(encoder.state_dict(), save_path)
    print(f"\n✓ Encoder saved to {save_path}")
    return encoder


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining for LOB encoder"
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/train.parquet',
        help='Path to training parquet file'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of pretraining epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (CPU-safe)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Training device'
    )

    parser.add_argument(
        '--save',
        type=str,
        default='models/encoder_pretrained.pt',
        help='Path to save encoder weights'
    )

    args = parser.parse_args()

    train_encoder(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save
    )

