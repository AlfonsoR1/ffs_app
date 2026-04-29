

## this file will collect the datasets needed for the ffs project
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
import requests

import nflreadpy as nfl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



def collect_data():
    """
    Collect recent NFL player data for fantasy football prediction work.
    """

    years = [2025]

    weekly_data = nfl.load_player_stats(years).to_pandas()

    print("Available columns:")
    print(weekly_data.columns.tolist())

    return weekly_data


def save_data():
    """
    Save NFL player data into a local data folder.
    """

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    weekly_data = collect_data()

    weekly_data.to_csv(
        os.path.join(data_folder, "weekly_player_data.csv"),
        index=False
    )

    print("Weekly player data saved.")
    print("Rows and columns:", weekly_data.shape)
    print(weekly_data.head())


if __name__ == "__main__":
    save_data()