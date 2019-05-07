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


# Function to load data, return final and opening prices and volume for each day
# convert data to correct format and store in arrays
def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name, skiprows=0, nrows=num_data_points, usecols=['Price', 'Open', 'Vol.'])

    # Prices of stock at the end of each day
    final_price = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    # Prices of stock at the beginning of each day
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    # Volume of stock exchanged throughout the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_price, opening_prices, volumes


# calculating price differences
def calculate_price_diff(final_price, opening_prices):
    price_diff = []
    for d_i in range(len(final_price) - 1):
        # difference between next day's opening price and current day's final price
        price_diff1 = opening_prices[d_i + 1] - final_price[d_i]
        price_diff.append(price_diff1)
    return price_diff


# calculating percentage of accuracy
def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values [a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values [a_i]:
            num_correct += 1
    return (num_correct/ len(actual_values)) * 100


# Training data sets
train_final_prices, train_opening_prices, train_volumes = load_stock_data(current_train_data, NUM_TRAIN_DATA_POINTS)
train_price_differences = calculate_price_diff(train_final_prices, train_opening_prices)
train_volumes = train_volumes[:-1]

# Testing data sets
test_final_prices, test_opening_prices, test_volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
test_price_differences = calculate_price_diff(test_final_prices, test_opening_prices)
test_volumes = test_volumes[:-1]



finals, openings, volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
diff = calculate_price_diff(finals, openings)
print(diff)