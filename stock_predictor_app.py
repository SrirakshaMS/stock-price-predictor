# Import libraries
from calendar import c
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Define date range
start_date = '2010-01-01'
end_date = '2023-10-04'

# Loading Data
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start_date, end=end_date)
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')
df.head(5)

# Describing Data
st.subheader('Data from Jan 2010 - Oct 2021')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
MA100 = df.Close.rolling(window=100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Original Price')
plt.plot(MA100, label='MA100')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
MA100 = df.Close.rolling(window=100).mean()
MA200 = df.Close.rolling(window=200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Original Price')
plt.plot(MA100, 'r', label='MA100')
plt.plot(MA200, 'g', label='MA200')
plt.legend()
st.pyplot(fig)

# Splitting into training and testing data into 70:30 ratio
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Scale down dataset i.e, normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Load pre-trained LSTM Model
model = load_model('stock_predictor.h5')

# Testing
past_100_days = pd.DataFrame(data_training.tail(100))
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(X_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot Original Price and Predicted Price
st.subheader('Predicted Price vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


