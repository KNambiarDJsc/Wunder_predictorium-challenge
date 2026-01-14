"""
Supervised predictor training script (FAST + SAFE).
CRITICAL: Last timestep only, small batches, no shuffle, two-phase training
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from src.dataset import SequenceDataset
from src.encoder import TemporalEncoder
from src.model import Predictor
from src.loss import CompetitionLoss


def train_predictor(
    data_path='data/train.parquet',
    encoder_path='models/encoder_pretrained.pt',
    epochs_phase1=35,   # 🔥 reduced (was 80)
    epochs_phase2=15,   # 🔥 reduced (was 40)
    batch_size=4,       # 🔥 faster on CPU than 8
    lr_phase1=5e-4,
    lr_phase2=5e-5,
    device='cpu',
    save_path='models/predictor_final.pt',
    resume=True
):
    device = torch.device(device)
    print(f"Training on {device}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ===================== DATA =====================
    print("Loading dataset...")
    dataset = SequenceDataset(data_path, store_numpy=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # CRITICAL
        num_workers=2,          # Windows-friendly
        pin_memory=False        # CPU safe
    )
    print(f"Loaded {len(dataset)} sequences")

    # ===================== MODEL =====================
    print("Loading pretrained encoder...")
    encoder = TemporalEncoder(d_in=64, d_hidden=64, d_gru=96)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    model = Predictor(encoder, freeze_encoder=True, unfreeze_gru=False).to(device)

    criterion = CompetitionLoss(
        alpha_mse=0.15,
        alpha_conf=0.1,
        alpha_abstain=0.05
    )

    # ===================== PHASE 1 =====================
    print("\n" + "=" * 60)
    print("PHASE 1: Encoder frozen")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_phase1,
        eta_min=lr_phase1 * 0.1
    )

    best_corr = -1.0
    start_epoch = 0
    ckpt_path = save_path.replace('.pt', '_phase1_ckpt.pt')

    # -------- Resume Phase 1 --------
    if resume and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_corr = ckpt["best_corr"]
        print(f"✓ Resumed Phase 1 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs_phase1):
        model.train()

        loss_sum = 0.0
        corr_sum = 0.0
        conf_sum = 0.0
        steps = 0

        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask = batch['pred_mask'].to(device)

            optimizer.zero_grad()

            # 🔥 FAST PATH: last timestep only
            pred, conf = model(x[:, -1:, :])

            y_last = y[:, -1, :]
            mask_last = mask[:, -1]

            loss, metrics = criterion(pred, conf, y_last, mask_last)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            corr_sum += 0.5 * (metrics['corr_t0'] + metrics['corr_t1'])
            conf_sum += metrics['mean_conf']
            steps += 1

        scheduler.step()

        mean_loss = loss_sum / steps
        mean_corr = corr_sum / steps
        mean_conf = conf_sum / steps

        print(
            f"Epoch {epoch+1}/{epochs_phase1} | "
            f"Loss {mean_loss:.4f} | "
            f"Corr {mean_corr:.4f} | "
            f"Conf {mean_conf:.3f} | "
            f"Abstain {(1-mean_conf):.1%}"
        )

        # Save best + checkpoint
        if mean_corr > best_corr:
            best_corr = mean_corr
            torch.save(model.state_dict(), save_path.replace('.pt', '_phase1_best.pt'))

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_corr": best_corr
        }, ckpt_path)

    # ===================== PHASE 2 =====================
    print("\n" + "=" * 60)
    print("PHASE 2: GRU unfrozen")
    print("=" * 60)

    model.load_state_dict(torch.load(save_path.replace('.pt', '_phase1_best.pt')))

    for p in model.encoder.gru.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase2,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_phase2,
        eta_min=lr_phase2 * 0.1
    )

    for epoch in range(epochs_phase2):
        model.train()

        loss_sum = 0.0
        corr_sum = 0.0
        conf_sum = 0.0
        steps = 0

        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask = batch['pred_mask'].to(device)

            optimizer.zero_grad()

            pred, conf = model(x[:, -1:, :])
            y_last = y[:, -1, :]
            mask_last = mask[:, -1]

            loss, metrics = criterion(pred, conf, y_last, mask_last)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            corr_sum += 0.5 * (metrics['corr_t0'] + metrics['corr_t1'])
            conf_sum += metrics['mean_conf']
            steps += 1

        scheduler.step()

        mean_loss = loss_sum / steps
        mean_corr = corr_sum / steps
        mean_conf = conf_sum / steps

        print(
            f"[P2] Epoch {epoch+1}/{epochs_phase2} | "
            f"Loss {mean_loss:.4f} | "
            f"Corr {mean_corr:.4f} | "
            f"Conf {mean_conf:.3f}"
        )

        if mean_corr > best_corr:
            best_corr = mean_corr
            torch.save(model.state_dict(), save_path)

    print(f"\n✓ Training complete | Best corr = {best_corr:.4f}")
    print(f"✓ Model saved to {save_path}")
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train.parquet')
    parser.add_argument('--encoder', type=str, default='models/encoder_pretrained.pt')
    parser.add_argument('--epochs-p1', type=int, default=35)
    parser.add_argument('--epochs-p2', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr-p1', type=float, default=5e-4)
    parser.add_argument('--lr-p2', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save', type=str, default='models/predictor_final.pt')

    args = parser.parse_args()

    train_predictor(
        data_path=args.data,
        encoder_path=args.encoder,
        epochs_phase1=args.epochs_p1,
        epochs_phase2=args.epochs_p2,
        batch_size=args.batch_size,
        lr_phase1=args.lr_p1,
        lr_phase2=args.lr_p2,
        device=args.device,
        save_path=args.save
    )
