from numpy import split
import streamlit as st
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt  # data-visualization
from streamlit_option_menu import option_menu
#%matplotlib inline
import seaborn as sns  # built on top of matplotlib
sns.set()
import pandas as pd  # working with data frames
import plotly.express as px
import numpy as np  # scientific computing
import missingno as msno  # analysing missing data
#import tensorflow as tf  # used to train deep neural network architectures
import tensorflow as tf  # used to train deep neural network architectures
from tensorflow.python.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.models import load_model
st.set_page_config(page_title='POWERTELL',page_icon=':bar_chart:',layout='wide')
st.title("  POWERTELL ")
st.markdown("Predicts any dataset")
st.sidebar.subheader("Effective solar production using LSTM")
#st.image("C:/Users/Akhila Rose Sabu/Desktop/Ashwin/solar.jpg")
data=st.file_uploader('upload a file')
df=pd.read_csv(data)

d=st.number_input("Enter the datatime column number")
d=round(d)
j=st.number_input("Enter the solar production column number")
j=round(j)
#country=st.text_input("Enter the country name ")
time=st.number_input("Time interval (Hours) between samples")
unit=st.selectbox('Enter the unit',['MWh','KWh', 'Wh'])
price=st.number_input("Enter the energy price(in USD) per unit of the region in your dataset")

name=df.columns[d]
solar=df.columns[j]
st.write("selected column name : {}".format(solar))
df=df.fillna(0)
df[name]=pd.to_datetime(df[name])
#df[name]=df[name].astype(np.datetime64)
df.set_index(name, inplace=True)  # set the datetime columns to be the index
df.index.name = "datetime"  # change the name of the index
#st.dataframe(df)
df=df.iloc[:,j-1:j]
#st.dataframe(df)
a=round(len(df.index)*0.2)
b = a
#test_data=df[[solar]].copy()
test_data=df.copy()
test_data=test_data[-a:]
#st.dataframe(test_data)
train_df,test_df=df[:-a],df[-a:]
st.success('Success message')
video_file = open('sunny.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
train = train_df
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_reshape = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_reshape=np.reshape(s_reshape,len(s_reshape))
    scalers['scaler_'+ i] = scaler
    train[i]=s_reshape
test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_reshape = scaler.transform(test[i].values.reshape(-1,1))
    s_reshape=np.reshape(s_reshape,len(s_reshape))
    scalers['scaler_'+i] = scaler
    test[i]=s_reshape
def split_series(series, past, future):
  
  X, y = list(), list()
  for window_start in range(len(series)):
    past_ends = window_start + past
    future_ends = past_ends + future
    if future_ends > len(series):
      break
    past1, future1 = series[window_start:past_ends, :], series[past_ends:future_ends, :]
    X.append(past1)
    y.append(future1)
  return np.array(X), np.array(y)

past = 720
future =24
features = 1
X_train, y_train = split_series(train_df.values, past, future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], features))
X_test, y_test = split_series(test_df.values,past, future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], features))
news=tf.keras.models.load_model("presentation7.h5")

prediction_model=news.predict(X_test)

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]
    prediction_model[:,:,index]=scaler.inverse_transform(prediction_model[:,:,index])
  #pred_e2d2[:,:,index]=scaler.inverse_transform(pred_e2d2[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

from sklearn.metrics import mean_absolute_error
for index,i in enumerate(train_df.columns):
    print(i)
    for j in range(1,2):
      print("Hour ",j,":")
      print("MAE-E1D1 : ",mean_absolute_error(y_test[:,j-1,index],prediction_model[:,j-1,index]))
    print()
x = len(prediction_model)
c=[]
for i in range(0,x):
  a=prediction_model[i][0][0]
  c.append(a)  
c = [0 if ele<0 else ele for ele in c]
st.success('ON THE WAY..... ')
test_data = test_data.reset_index()
#st.markdown(solar)
new=test_data.iloc[767:,1]

# plot the ground-truth and forecast and compare them with residuals
sns.set_context("poster")
fig,(ax3) = plt.subplots(1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
sns.lineplot(x = test_data.datetime[767:], y=new, color="yellow", ax=ax3, label="original") # get the ground-truth validation data
sns.lineplot(x = test_data.datetime[767:], y=c[:-24], color="blue", dashes=True, ax=ax3, label="Forecast", alpha=0.4)  # get the forecast
# set the axis labels and title
ax3.set_xlabel("Date")
ax3.set_ylabel("Solar Generation (MW)")
ax3.set_title("Time Series Data");  
# #plot for residual
# residuals = (new- c[:-24])
# sns.lineplot(y=residuals, x=test_data.datetime[767:], ax=ax4, label="Residuals")
# ax4.set_ylabel("Residuals"); # set the y-label for residuals
st.pyplot(fig)
distribution=sum(c[-24:])
if unit=='MWh':
     tt=round(distribution*1000)
elif unit=='Wh' :
     tt=round(distribution/1000)
else :
     tt=distribution
price_prediction=round(distribution*price,2)
hour=time*24
#prediction=price_prediction*hour
#st.write('We are going to predict 24 samples. Since your sample size is {} in hours, and the cost of electricity(in USD) {} per {}, you would get a profit of {} USD  in {} hours if you employ solar power units to harvest energy.'.format(time,price,unit,price_prediction,hour))
st.write("Annual energy costs can be in the thousands. The average annual energy expenditure per person is $3,052, including transportation and residential energy. Solar power can reduce or eliminate these costs as soon as they are installed. They also offer long-term savings, because it’s basically free to capture the power of the sun. Solar panels are a great way to offset energy costs and reduce the environmental impact. Traditional electricity is sourced from fossil fuels such as coal and natural gas. When fossil fuels are burned to produce electricity, they emit harmful gases that are the primary cause of air pollution and global climate change. Other than these impacts, they are finite resources and can cause fluctuations in energy prices.")
st.write("Implementing solar panels and harvesting solar power is the best option to tackle this problem. The government provides about 30%  of the implementation cost as a subsidy to promote solar generation in the country. The solar panel is the superlative option for long-term runs and savings. The industry standard for most solar panel’s lifespans is 25 to 30 years. Most reputable manufacturers offer production warranties for 25 years or more. The owner can start earning from the first day.")
st.write("On average, a one-meter square solar panel can harvest about 1000 watts of sunlight in a day. The price for 1000 {} is {} USD  According to the dataset provided, an amount of ${} can be generated in {} Hours, if {} sq.m solar panels are implemented. This exhibit that, the implementation cost can be fully covered in the initial 4 years. From the 5th year onwards, panels generate profits with very low maintenance costs.".format(unit,price,price_prediction,hour,tt))
st.balloons()
st.snow()
st.balloons()
