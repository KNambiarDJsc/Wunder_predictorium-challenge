"""
Supervised predictor training script.
CRITICAL: Last timestep only, small batches, no shuffle, two-phase training
THIS IS WHERE MOST PEOPLE FAIL
"""
import torch
import torch.nn as nn
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
    epochs_phase1=80,
    epochs_phase2=40,
    batch_size=8,  # CRITICAL: ≤ 8
    lr_phase1=5e-4,
    lr_phase2=5e-5,
    device='cpu',
    save_path='models/predictor_final.pt'
):
    """
    Train supervised predictor with correlation-aware loss.
    
    CRITICAL REQUIREMENTS:
    - Train on LAST timestep ONLY: pred[:, -1]
    - Batch size ≤ 8 (smaller = better correlation gradients)
    - NO SHUFFLE (temporal integrity)
    - Gradient clipping EVERY step
    - Cosine LR schedule
    - Two-phase training:
        Phase 1: encoder frozen
        Phase 2: unfreeze GRU only, LR × 0.1
    - Track abstention rate (target: 70-85% silence)
    """
    device = torch.device(device)
    print(f"Training on {device}")
    
    # Create models directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Data (NO SHUFFLE, batch size ≤ 8)
    print("Loading dataset...")
    dataset = SequenceDataset(data_path, store_numpy=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # CRITICAL: ≤ 8
        shuffle=False,  # CRITICAL: NO SHUFFLE
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Loaded {len(dataset)} sequences")
    
    # Load pretrained encoder
    print("Loading pretrained encoder...")
    encoder = TemporalEncoder(d_in=64, d_hidden=64, d_gru=96)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    
    # Model (encoder frozen initially)
    model = Predictor(encoder, freeze_encoder=True, unfreeze_gru=False).to(device)
    
    # Loss
    criterion = CompetitionLoss(alpha_mse=0.15, alpha_conf=0.1, alpha_abstain=0.05)
    
    # ========== PHASE 1: ENCODER FROZEN ==========
    print("\n" + "="*60)
    print("PHASE 1: Training with frozen encoder")
    print("="*60)
    
    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1,
        weight_decay=1e-5
    )
    
    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_phase1,
        eta_min=lr_phase1 * 0.1
    )
    
    best_corr = -1.0
    
    for epoch in range(epochs_phase1):
        model.train()
        
        epoch_losses = []
        epoch_corrs_t0 = []
        epoch_corrs_t1 = []
        epoch_confs = []
        
        for batch_idx, batch in enumerate(loader):
            x = batch['x'].to(device)  # (B, 1000, 64)
            y = batch['y'].to(device)  # (B, 1000, 2)
            mask = batch['pred_mask'].to(device)  # (B, 1000)
            
            optimizer.zero_grad()
            
            # CRITICAL: Get predictions for LAST timestep ONLY
            # Model.forward() already returns last timestep
            pred, conf = model(x)  # (B, 2), (B, 1)
            
            # Extract last timestep target
            y_last = y[:, -1, :]  # (B, 2)
            mask_last = mask[:, -1]  # (B,)
            
            # Compute loss on last timestep
            loss, metrics = criterion(pred, conf, y_last, mask_last)
            
            # Backward with gradient clipping EVERY step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_corrs_t0.append(metrics['corr_t0'])
            epoch_corrs_t1.append(metrics['corr_t1'])
            epoch_confs.append(metrics['mean_conf'])
        
        scheduler.step()
        
        # Statistics
        mean_loss = np.mean(epoch_losses)
        mean_corr_t0 = np.mean(epoch_corrs_t0)
        mean_corr_t1 = np.mean(epoch_corrs_t1)
        mean_corr = (mean_corr_t0 + mean_corr_t1) / 2
        mean_conf = np.mean(epoch_confs)
        
        # TRACK ABSTENTION RATE (target: 70-85% silence)
        abstention_rate = 1.0 - mean_conf
        
        print(f"Epoch {epoch+1}/{epochs_phase1} | "
              f"Loss: {mean_loss:.4f} | "
              f"Corr: {mean_corr:.4f} (t0={mean_corr_t0:.4f}, t1={mean_corr_t1:.4f}) | "
              f"Conf: {mean_conf:.3f} | "
              f"Abstain: {abstention_rate:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if mean_corr > best_corr:
            best_corr = mean_corr
            torch.save(model.state_dict(), save_path.replace('.pt', '_phase1_best.pt'))
    
    # ========== PHASE 2: UNFREEZE GRU ONLY ==========
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with GRU unfrozen")
    print("="*60)
    
    # Load best from phase 1
    model.load_state_dict(torch.load(save_path.replace('.pt', '_phase1_best.pt')))
    
    # Unfreeze GRU only
    for param in model.encoder.gru.parameters():
        param.requires_grad = True
    
    # New optimizer with lower LR
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
        
        epoch_losses = []
        epoch_corrs_t0 = []
        epoch_corrs_t1 = []
        epoch_confs = []
        
        for batch_idx, batch in enumerate(loader):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask = batch['pred_mask'].to(device)
            
            optimizer.zero_grad()
            
            pred, conf = model(x)
            y_last = y[:, -1, :]
            mask_last = mask[:, -1]
            
            loss, metrics = criterion(pred, conf, y_last, mask_last)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_corrs_t0.append(metrics['corr_t0'])
            epoch_corrs_t1.append(metrics['corr_t1'])
            epoch_confs.append(metrics['mean_conf'])
        
        scheduler.step()
        
        mean_loss = np.mean(epoch_losses)
        mean_corr_t0 = np.mean(epoch_corrs_t0)
        mean_corr_t1 = np.mean(epoch_corrs_t1)
        mean_corr = (mean_corr_t0 + mean_corr_t1) / 2
        mean_conf = np.mean(epoch_confs)
        abstention_rate = 1.0 - mean_conf
        
        print(f"Epoch {epoch+1}/{epochs_phase2} | "
              f"Loss: {mean_loss:.4f} | "
              f"Corr: {mean_corr:.4f} (t0={mean_corr_t0:.4f}, t1={mean_corr_t1:.4f}) | "
              f"Conf: {mean_conf:.3f} | "
              f"Abstain: {abstention_rate:.1%} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if mean_corr > best_corr:
            best_corr = mean_corr
            torch.save(model.state_dict(), save_path)
    
    print(f"\n✓ Best correlation: {best_corr:.4f}")
    print(f"✓ Model saved to {save_path}")
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train.parquet')
    parser.add_argument('--encoder', type=str, default='models/encoder_pretrained.pt')
    parser.add_argument('--epochs-p1', type=int, default=80)
    parser.add_argument('--epochs-p2', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
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