import json
import pandas as pd


def get_data(path):
    return pd.read_csv(path)
    
def get_params(path):
    with open(path, "r") as f:
        return json.load(f)