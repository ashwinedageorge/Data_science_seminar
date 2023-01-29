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

st.set_page_config(page_title='MODEL',page_icon=':bar_chart:',layout='wide')
st.title("MODEL FOR OUR DATASET")

#data=st.file_uploader('upload a file')
# df=pd.read_csv("C:/Users/Akhila Rose Sabu/Downloads/time_series_15min_singleindex (1).csv")
# df["utc_timestamp"] = df["utc_timestamp"].astype(np.datetime64)  # set the data type of the datetime column to np.datetime64
# df.set_index("utc_timestamp", inplace=True)  # set the datetime columns to be the index
# df.index.name = "datetime"  # change the name of the index
# df=df.fillna(0)
country=option_menu('Country',['None','Austria', 'Deutschland', 'Deutschland_50hertz', 'Deutschland_LU', 'Deutschland_amprion', 'Deutschland_tennet', 'Deutschland_transnetbw',  'Netherlands'], default_index=0)
if (country == 'None'):
    st.warning('Please select a country')

#1
if (country == 'Austria'):
    st.image("austria.png")

#2
if (country == 'Deutschland'):
    st.image("deutc.png")

#3    
if (country == 'Deutschland_50hertz'):
     st.image("de5.png")

#4    
if (country == 'Deutschland_LU'):
     st.image("lu.png")
  


#5
if (country == 'Deutschland_amprion'):
      st.image("Amprion.png")
     


#6
if (country == 'Deutschland_tennet'):
       st.image("tennt.png")  

#7
if (country == 'Deutschland_transnetbw'):
       st.image("transt.png")


#8
if (country == 'Netherlands'):
       st.image("nethrlnd.png")