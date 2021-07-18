#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
動画
#Stock Price Prediction Using Python & Machine Learning
#https://www.youtube.com/watch?v=QIUxPv5PJOY&t=2114s

記事
https://randerson112358.medium.com/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
'''


# In[6]:


'''
# Description
This program uses an artificial recurrent neural network 
called Long Short Term Memory (LSTM) 
to predict the closing stock price of a corporation (Apple Inc.) 
using the past 60 day stock price.
このプログラムの内容
このブログラムはLSTMと呼ばれる人工リカレントニューラルネットワークを使って
過去60日の株価からアップル社の株価の終値を予測します

'''


# In[1]:


#必要なライブラリのインポート
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime
plt.style.use('fivethirtyeight')


# In[2]:


#Yahoo financeからアップル社の株価のデータを引用する
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17') 
#Show the data 
df.head()


# In[3]:


#dfの行と列の数を表示する。（インデックス行頭は含まれない）
#2003の行、6の列があることを表す。
df.shape


# In[4]:


#終値をグラフ化する
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()


# In[5]:


#dfから終値だけのデータフレームを作成する。（インデックスは残っている）
data = df.filter(['Close'])
#dataをデータフレームから配列に変換する。
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.values.html
dataset = data.values
#トレーニングデータを格納するための変数を作る。
#データの80%をトレーニングデータとして使用する。今回は2003のうち1603をトレーニングに使用する
#math.ceil()は小数点以下の切り上げ
#https://note.nkmk.me/python-math-floor-ceil-int/
training_data_len = math.ceil(len(dataset) * .8)


# In[6]:


#データセットを0から1までの値にスケーリングする。
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
scaler = MinMaxScaler(feature_range=(0, 1)) 
#fitは変換式を計算する
#transform は fit の結果を使って、実際にデータを変換する
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler.fit_transform
scaled_data = scaler.fit_transform(dataset)
#fit,transfrom,fit_transformの違い
#https://mathwords.net/fittransform
#これを行うのは、データをニューラルネットワークに渡す前にスケーリングするのが一般的に良い習慣だからです。


# In[7]:


'''
61番目の終値を予測するため、過去60日間の終値を含むトレーニングデータセットを作成します。

x_trainデータセットの最初の列には、インデックス0からインデックス59までのデータセットの値（合計60個の値）が含まれ、
2番目の列にはインデックス1からインデックス60までのデータセットの値（60個の値）が含まれます。

y_trainデータセットには、最初の列のインデックス60にある61番目の値と、
2番目の値のデータセットのインデックス61にある62番目の値が含まれます。
'''


# In[8]:


len(scaled_data)


# In[9]:


#正規化されたデータセットを作る。データ数はトレーニングデータ数にする
train_data = scaled_data[0:training_data_len, :]
len(train_data)


# In[10]:


#データをx_trainとy_trainのセットに分ける
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#[i:j]の場合、j-1までが抽出される


# In[11]:


'''
x_train = (array([60コ]), array([60コ]), ・・・)と60コの行データが1543コ入っている
y_train = (a,b,c,・・・)と1543コのデータが入っている
'''


# In[12]:


x_train


# In[13]:


len(y_train)


# In[14]:


'''
独立トレーニングデータセット「x_train」と従属トレーニングデータセット「y_train」をnumpy配列に変換して、
LSTMモデルのトレーニングに使用できるようにします。
'''


# In[15]:


x_train, y_train = np.array(x_train), np.array(y_train)


# In[16]:


'''
x_train = array([[a,b,c,・・・],
                [d,e,f,・・・],
                ・・・])と60列のデータが1543行入っている。
y_train = array([a,b,c,・・・])と1543列のデータが1行に入っている。
'''


# In[17]:


x_train


# In[18]:


y_train


# In[19]:


x_train.shape[0] #行数


# In[20]:


x_train.shape[1] #列数


# In[21]:


'''
データを[サンプル数、タイムステップ数、および特徴数]の形式で3次元に再形成します。
LSTMモデルは、3次元のデータセットを想定しています。
'''


# In[22]:


#LSTMに受け入れられる形にデータをつくりかえます
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[23]:


'''
https://note.nkmk.me/python-numpy-reshape-usage/
numpy.reshape(numpy.ndarray,(a, b, c))
データをa個のndarrayに分ける
その分けられたデータはb行、c列の形状になる

x_train = array([[a],
                [b],
                [c],
                ・
                ・
                ・]) と60行1列のデータを1543コ作る
'''


# In[24]:


x_train[1]


# In[ ]:





# In[25]:


#LSTMモデルを構築して、50ニューロンの2つのLSTMレイヤーと2つの高密度レイヤーを作成します。（1つは25ニューロン、もう1つは1ニューロン）
#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


# In[27]:


#平均二乗誤差（MSE）損失関数とadamオプティマイザーを使用してモデルをコンパイルします。
model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


'''
トレーニングデータセットを使用してモデルをトレーニングします。
フィットはtrainの別名であることに注意してください。
バッチサイズは、単一のバッチに存在するトレーニング例の総数であり、
エポックは、データセット全体がニューラルネットワークを前後に渡されるときの反復回数です。
'''


# In[28]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[29]:


#Test data set
#test_dataはtraining_dataから60コ減らしたデータ
#つまり60日間の終値から将来の終値を予測するためにtest_dataを用意する
test_data = scaled_data[training_data_len - 60: , : ]


# In[31]:


training_data_len 


# In[33]:


len(test_data)


# In[54]:


#Create the x_test and y_test data sets
#1603行目以降のデータをすべてy_testに入れる (この場合は終値だけ), 
#つまり 2003 - 1603 = 400 行のデータになる
x_test = [] #予測値をつくるために使用するデータを入れる
y_test =  dataset[training_data_len : , : ] #実際の終値データ

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0]) #正規化されたデータ


# In[52]:


y_test


# In[50]:


test_data[399:459,0]


# In[ ]:


'''
次に、独立したテストデータセット「x_test」をnumpy配列に変換して、
LSTMモデルのテストに使用できるようにします。
'''


# In[48]:


x_test[399:]


# In[55]:


#Convert x_test to a numpy array 
x_test = np.array(x_test)
#60列のデータが400行入ったものになる


# In[57]:


x_test.shape


# In[40]:


'''
Reshape the data to be 3-dimensional in the form [number of samples, number of time steps, and number of features].
This needs to be done, because the LSTM model is expecting a 3-dimensional data set.
'''


# In[58]:


#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#60行1列のデータが400コ入った形に変換する


# In[61]:


x_test.shape


# In[ ]:


#Now get the predicted values from the model using the test data.


# In[62]:


#Getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #Undo scaling


# In[ ]:


'''
Get the root mean squared error (RMSE), 
which is a good measure of how accurate the model is. 
A value of 0 would indicate that the models predicted values match the actual values from the test data set perfectly.

The lower the value the better the model performed. 
But usually it is best to use other metrics as well to truly get an idea of how well the model performed.
'''


# In[63]:


#Calculate/Get the value of RMSE
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse


# In[64]:


#Let’s plot and visualize the data.
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[65]:


#Show the valid and predicted prices
valid


# In[ ]:


'''
I want to test the model some more 
and get the predicted closing price value of Apple Inc. for December 18, 2019 (12/18/2019).

So I will get the quote, convert the data to an array 
that contains only the closing price.
Then I will get the last 60 day closing price
and scale the data to be values between 0 and 1 inclusive.

After that I will create an empty list and append the past 60 day price to it, 
and then convert it to a numpy array 
and reshape it 
so that I can input the data into the model.

Last but not least, 
I will input the data into the model and get the predicted price.
'''


# In[66]:


#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get teh last 60 day closing price 
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[68]:


#Now let’s see what the actual price for that day was.

#Get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




