"""
Dataset with proper sequence handling and no leakage.
CRITICAL: Train = Inference, No future peeking
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import compute_features


class SequenceDataset(Dataset):
    """
    Loads LOB sequences with proper temporal integrity.
    NO LOSS COMPUTATION - dataset only provides data + mask.
    """
    def __init__(self, parquet_path, store_numpy=True):
        """
        Args:
            parquet_path: path to parquet file
            store_numpy: if True, store as numpy and convert in __getitem__
        """
        df = pd.read_parquet(parquet_path)
        
        # Feature columns
        price_cols = [f'p{i}' for i in range(12)]
        vol_cols = [f'v{i}' for i in range(12)]
        dp_cols = [f'dp{i}' for i in range(4)]
        dv_cols = [f'dv{i}' for i in range(4)]
        target_cols = ['t0', 't1']
        
        self.sequences = []
        self.store_numpy = store_numpy
        
        for seq_ix, group in df.groupby('seq_ix'):
            # CRITICAL: Sort by step_in_seq
            group = group.sort_values('step_in_seq').reset_index(drop=True)
            
            # ASSERT sequence length
            assert len(group) == 1000, f"Sequence {seq_ix} has {len(group)} steps, expected 1000"
            
            # Extract raw arrays
            p = group[price_cols].values.astype(np.float32)
            v = group[vol_cols].values.astype(np.float32)
            dp = group[dp_cols].values.astype(np.float32)
            dv = group[dv_cols].values.astype(np.float32)
            
            # Engineer features
            x = compute_features(p, v, dp, dv)  # (1000, 64)
            
            # Extract targets
            y = group[target_cols].values.astype(np.float32)  # (1000, 2)
            
            # CRITICAL: BOOLEAN MASK (not float)
            # pred_mask = step_in_seq >= 99
            pred_mask = (group['step_in_seq'].values >= 99)
            assert pred_mask.dtype == bool, "Mask must be boolean"
            
            if store_numpy:
                # Store as numpy, convert to torch in __getitem__
                self.sequences.append({
                    'x': x,  # numpy
                    'y': y,  # numpy
                    'pred_mask': pred_mask,  # numpy bool
                    'seq_ix': seq_ix
                })
            else:
                self.sequences.append({
                    'x': torch.from_numpy(x),
                    'y': torch.from_numpy(y),
                    'pred_mask': torch.from_numpy(pred_mask),
                    'seq_ix': seq_ix
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns single sequence.
        NO LOSS COMPUTATION HERE - training loop enforces warm-up exclusion.
        """
        seq = self.sequences[idx]
        
        if self.store_numpy:
            # DELAY TORCH CONVERSION to __getitem__
            return {
                'x': torch.from_numpy(seq['x']),
                'y': torch.from_numpy(seq['y']),
                'pred_mask': torch.from_numpy(seq['pred_mask']),
                'seq_ix': seq['seq_ix']
            }
        else:
            return seq


class StreamingDataset:
    """
    For inference - processes data points one at a time with state.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset for new sequence."""
        self.buffer = []
        self.step = 0
    
    def add_point(self, p, v, dp, dv):
        """
        Add single data point and compute features.
        
        Returns:
            features: (64,) numpy array
        """
        # Use batch feature computation on accumulated buffer
        # More efficient than per-step computation
        temp_p = np.array([p], dtype=np.float32)
        temp_v = np.array([v], dtype=np.float32)
        temp_dp = np.array([dp], dtype=np.float32)
        temp_dv = np.array([dv], dtype=np.float32)
        
        # If we have history, need to compute with context
        if len(self.buffer) > 0:
            # Append to buffer and recompute (inefficient but correct)
            all_p = np.vstack([b[0] for b in self.buffer] + [temp_p])
            all_v = np.vstack([b[1] for b in self.buffer] + [temp_v])
            all_dp = np.vstack([b[2] for b in self.buffer] + [temp_dp])
            all_dv = np.vstack([b[3] for b in self.buffer] + [temp_dv])
            
            features = compute_features(all_p, all_v, all_dp, all_dv)
            current_features = features[-1]  # Last timestep
        else:
            # First point
            features = compute_features(temp_p, temp_v, temp_dp, temp_dv)
            current_features = features[0]
        
        self.buffer.append((temp_p, temp_v, temp_dp, temp_dv))
        self.step += 1
        
        return current_features
    
    def get_sequence(self):
        """
        Get full sequence up to current step.
        
        Returns:
            x: (T, 64) numpy array
        """
        if len(self.buffer) == 0:
            return np.zeros((0, 64), dtype=np.float32)
        
        # Reconstruct full sequence
        all_p = np.vstack([b[0] for b in self.buffer])
        all_v = np.vstack([b[1] for b in self.buffer])
        all_dp = np.vstack([b[2] for b in self.buffer])
        all_dv = np.vstack([b[3] for b in self.buffer])
        
        return compute_features(all_p, all_v, all_dp, all_dv)