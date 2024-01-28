# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:15:10 2024

@author: shiva
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
 

model4 = load_model('my_model.keras')


st.title('Stock Price Prediction')
user_input = st.text_input('Enter Stock Ticker', 'RELIANCE.NS')

start='2000-01-01'
end='2024-01-05'
data=yf.download(user_input,start,end)

st.subheader('Stock data')
st.write(data)

st.subheader('Description of data')
st.write(data.describe())



# splitting date into training and testing 
data_train= pd.DataFrame(data.Close[0:int(len(data)*0.86)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.86):len(data)])




# scaling of data using min max scaler (0,1)Visualiztion of v
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days=data_train.tail(100)
data_test=pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scaler=scaler.fit_transform(data_test)



st.subheader('MA30 vs Close price')
ma_30_days=data.Close.rolling(30).mean()
fig1= plt.figure(figsize = (12,6))
plt.plot(ma_30_days, 'b', label = 'MA30 DAYS')
plt.plot(data.Close, 'r', label = 'Close price')
plt.legend()
st.pyplot(fig1)

st.subheader('MA30 vs MA100 vs Close price')
ma_100_days=data.Close.rolling(100).mean()
fig2= plt.figure(figsize = (12,6))
plt.plot(ma_30_days, 'b', label = 'MA30 DAYS')
plt.plot(ma_100_days, 'g', label = 'MA100 DAYS')
plt.plot(data.Close, 'r', label = 'Close price')
plt.legend()
st.pyplot(fig2)


x= []
y = []

for i in range (100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100 : i])
    y.append(data_test_scaler[i, 0])
x, y = np.array(x), np.array(y)




predict=model4.predict(x) 
scale=1/scaler.scale_
predict=predict*scale
y=y*scale



st.subheader('original price vs predicted price')
fig3=plt.figure(figsize = (12,6))
plt.plot(y, 'r', label = 'original Price')
plt.plot(predict, 'b', label = 'predicted Price')
plt.legend()
st.pyplot(fig3)