

## this file will collect the datasets needed for the ffs project
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
import requests

import nfl_data_py as nfl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def collect_data():
    # Collect play-by-play data for the 2025 season
    pbp_data = nfl.import_pbp_data(years=[2025])
    
    # Collect player data
    player_data = nfl.import_rosters(years=[2025])
    
    # Collect team data
    team_data = nfl.import_team_desc()
    
    # Collect weather data (if available)
    weather_data = nfl.import_weather_data(years=[2025])
    
    return pbp_data, player_data, team_data, weather_data

if __name__ == "__main__":
    pbp_data, player_data, team_data, weather_data = collect_data()
    
    # Save the collected data to CSV files for later use
    pbp_data.to_csv('pbp_data_2025.csv', index=False)
    player_data.to_csv('player_data_2025.csv', index=False)
    team_data.to_csv('team_data_2025.csv', index=False)
    weather_data.to_csv('weather_data_2025.csv', index=False)