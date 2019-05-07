import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt

GOLD_TRAIN_DATA = 'CSV Files/Gold Futures Historical 1 Year.csv'
GOLD_TEST_DATA = 'CSV Files/Gold Futures Historical Data.csv'

current_train_data = GOLD_TRAIN_DATA
current_test_data = GOLD_TEST_DATA

NUM_TRAIN_DATA_POINTS = 288
NUM_TEST_DATA_POINTS = 22


# Function to load data, convert to correct format and store in arrays
def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name, skiprows=0, nrows=num_data_points, usecols=['Price', 'Open', 'Vol.'])
    final_price = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_price, opening_prices, volumes

print(load_stock_data(current_test_data, NUM_TEST_DATA_POINTS))
