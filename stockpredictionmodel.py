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

LEARNING_RATE = 0.1
NUM_EPOCHS = 100


# Function to load data, return final and opening prices and volume for each day
# convert data to correct format and store in arrays
def load_stock_data(stock_name, num_data_points):
    data = pd.read_csv(stock_name,
                       skiprows=0,
                       nrows=num_data_points,
                       usecols=['Price', 'Open', 'Vol.'])
    # Prices of stock at the end of each day
    final_prices = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    # Prices of stock at the beginning of each day
    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    # Volume of stock exchanged throughout the day
    volumes = data['Vol.'].str.strip('MK').astype(np.float)
    return final_prices, opening_prices, volumes


# Function to calculate differences between opening price of the next day and final price of the current day
def calculate_price_diff(final_prices, opening_prices):
    price_differences = []
    for d_i in range(len(final_prices) - 1):
        price_difference = opening_prices[d_i + 1] - final_prices[d_i]
        price_differences.append(price_difference)
    return price_differences


def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100



# Training data sets
train_final_prices, train_opening_prices, train_volumes = load_stock_data(current_train_data, NUM_TRAIN_DATA_POINTS)
train_price_differences = calculate_price_diff(train_final_prices, train_opening_prices)

# Getting rid of last set of volumes
train_volumes = train_volumes[:-1]

# Testing Data Sets
test_final_prices, test_opening_prices, test_volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
test_price_differences = calculate_price_diff((test_final_prices, test_opening_prices))
test_volumes = test_volumes[:-1]

# Building computational Graph
# y = Wx + b
x = tf.placeholder(tf.float32, name='x')
W = tf.Variable([.1], name='W')
b = tf.Variable([.1], name='b')
y = W * x + b
y_predicted = tf.placeholder(tf.float32, name='y_predicted')

# minimise loss
loss = tf.reduce_sum(tf.square(y - y_predicted))

# optimise loss
optimiser = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())
for _i in range(NUM_EPOCHS):
    session.run(optimiser, feed_dict={x: train_volumes, y_predicted: train_price_differences})