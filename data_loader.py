####################################################################################################################
# % Code for the paper:
# % Improving urban water demand forecast using conformal prediction-based hybrid machine learning models
# % By Oluwabunmi Iwakin; Farrah Moazeni, PhD
# % Lehigh University, omi222@lehigh.edu, moazeni@lehigh.edu
####################################################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(csv_path):
    """
    Loads and preprocesses water demand data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(csv_path, parse_dates=True)

    # Encode categorical variables
    if 'Season' in df.columns:
        le = LabelEncoder()
        df['Season'] = le.fit_transform(df['Season'])

    # Convert from gallons to cubic meters if applicable
    if 'Total_Demand' in df.columns:
        df['Total_Demand'] = df['Total_Demand'] / 264.1722

    return df
