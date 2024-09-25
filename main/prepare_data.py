import logging
import pandas as pd
import json
from utils import load_data

config_path = "C:\\Users\\piotr\\Python\\rehabnet\\main\\config.json"

if __name__ == '__main__':
    
    config = json.load(open(config_path))
    logs_folder = config['logs_folder']
    raw_data_path = config['raw_data_path']
    processed_data_path = config['processed_data_path']
    
    df = load_data(raw_data_path, processed_data_path)
    values = df['label'].value_counts()
    print(values)
    