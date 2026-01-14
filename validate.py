"""
Validation script that EXACTLY matches Wunder scorer behavior.
NO thresholds. NO abstention. NO shortcuts.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import weighted_pearson_correlation


def evaluate_solution(solution_cls, data_path):
    """
    Evaluate solution exactly like the competition scorer.
    """

    df = pd.read_parquet(data_path)
    model = solution_cls()

    preds = []
    targets = []

    # Mock DataPoint (matches utils.DataPoint interface)
    class DataPoint:
        pass

    print("Running scorer-style validation...")

    for seq_ix, group in tqdm(df.groupby('seq_ix'), desc="Sequences"):
        group = group.sort_values('step_in_seq').reset_index(drop=True)

        for _, row in group.iterrows():
            dp = DataPoint()
            dp.seq_ix = row['seq_ix']
            dp.step_in_seq = row['step_in_seq']
            dp.need_prediction = row['step_in_seq'] >= 99

            for i in range(12):
                setattr(dp, f'p{i}', row[f'p{i}'])
                setattr(dp, f'v{i}', row[f'v{i}'])
            for i in range(4):
                setattr(dp, f'dp{i}', row[f'dp{i}'])
                setattr(dp, f'dv{i}', row[f'dv{i}'])

            pred = model.predict(dp)

            if dp.need_prediction:
                # scorer expects a prediction EVERY time
                assert pred is not None
                pred = np.clip(pred, -6, 6)

                target = np.array([row['t0'], row['t1']], dtype=np.float32)

                preds.append(pred)
                targets.append(target)

    preds = np.array(preds)
    targets = np.array(targets)

    # EXACT scorer metric
    corr_t0 = weighted_pearson_correlation(preds[:, 0], targets[:, 0])
    corr_t1 = weighted_pearson_correlation(preds[:, 1], targets[:, 1])
    corr_avg = 0.5 * (corr_t0 + corr_t1)

    print("\n" + "=" * 60)
    print(f"Correlation t0 : {corr_t0:.5f}")
    print(f"Correlation t1 : {corr_t1:.5f}")
    print(f"Correlation avg: {corr_avg:.5f}")
    print("=" * 60)

    return {
        "corr_t0": corr_t0,
        "corr_t1": corr_t1,
        "corr_avg": corr_avg
    }


if __name__ == "__main__":
    from solution import PredictionModel

    evaluate_solution(
        solution_cls=PredictionModel,
        data_path="data/valid.parquet"
    )
