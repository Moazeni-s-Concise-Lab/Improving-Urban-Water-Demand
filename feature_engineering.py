import numpy as np
import pandas as pd

def create_features(df: pd.DataFrame, window: int = 12):
    """
    Creates time-series features and labels from the dataset.

    Args:
        df (pd.DataFrame): Input data.
        window (int): Number of past timesteps to include as features.

    Returns:
        Tuple: Feature matrix, label vector
    """
    new_format = []
    present_demand = []

    for idx, row in enumerate(df.values[window:-2]):
        weather_feat = row[1:-1]
        past_demand = df['Total_Demand'][idx:idx + window]
        combined = np.concatenate([weather_feat, past_demand]).astype(np.float32)
        new_format.append(combined)
        present_demand.append(row[-1])

    return new_format, present_demand
