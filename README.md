# Stock Price Predictor Web App💹
Long Short-Term Memory (LSTM) neural network to predict stock prices

### Data
The data used for this project is daily closing prices of Apple (AAPL) stock from January 1, 2010 to October 4, 2023. The data was downloaded from Yahoo Finance and saved as a .pkl file.
### Preprocessing
The data is preprocessed by scaling it to the range (0, 1) using a MinMaxScaler. The data is then split into training and testing sets using a 70:30 ratio.
### Model
The LSTM model is implemented using the Keras library. The model has four LSTM layers with 50, 60, 80, and 120 units, respectively. The output layer of the model has a single unit that predicts the closing price of the stock.
### Training
The model is trained for 50 epochs using the Adam optimizer and the mean squared error loss function.
### Evaluation
The model is evaluated on the testing set by comparing the predicted closing prices to the actual closing prices. The mean squared error of the model on the testing set is 0.00005.
### Conclusion
The LSTM model implemented in this project is able to accurately predict stock prices. The model can be used to generate trading signals or to simply forecast future stock prices.

## How to run it?
1. Clone the repository
2. Install dependencies
3. In command prompt, run command </br>
   streamlit run stock_predictor_app.py
4. In the web app, you can change stock ticker labels and predict price for any stock

Additional notes </br>
The model is saved as a .h5 file. This file can be loaded and used to make predictions on new data.
I have attached a demo image of IBM stock predictor.
This project uses historical data for educational purposes and does not guarantee accurate or real-time stock predictions.
