import streamlit as st
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt  # data-visualization
#%matplotlib inline
import seaborn as sns  # built on top of matplotlib
sns.set()
import pandas as pd  # working with data frames
import plotly.express as px
import numpy as np  # scientific computing
import missingno as msno  # analysing missing data
import tensorflow as tf  # used to train deep neural network architectures
from tensorflow.python.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# df=pd.read_csv("C:/Users/Akhila Rose Sabu/Downloads/time_series_15min_singleindex (1).csv")
# df = df.fillna(0)

# # changing the type of datetime coulmn
# df["utc_timestamp"] = df["utc_timestamp"].astype(np.datetime64)  # set the data type of the datetime column to np.datetime64
# df.set_index("utc_timestamp", inplace=True)  # set the datetime columns to be the index
# df.index.name = "datetime"  # change the name of the index
# # dropping 'cet_cest_timestamp' column
# df.drop(['cet_cest_timestamp'], axis=1)
# #df.info()
#df.describe()
st.header("EDA ANALYSIS OF THE DATASET")
#separating solar related columns to another dataframe
# #finding the number of solar generating sites 
# solar_generation_cols = [col for col in df.columns if 'solar_generation' in col]
# print(solar_generation_cols)
# print(len(solar_generation_cols))
# #This dataset have the details of 9 different solar generating regions.
# #dataframe with solar details
# df_production =df[solar_generation_cols]
#production per month by different site
st.markdown("### 1.Average Monthly Solar Production from all the regions")
# df_production.groupby(df_production.index.month).mean().plot(kind="bar",xlabel='Month',ylabel='Energy in KWh',title="Average Monthly Solar Production from all the regions", figsize=(20,12))
# st.pyplot(plt.gcf())
st.image("one.jpg")
st.write("The average production of solar power over different months in the years 2015 - 2020 is shown here. Comparitively Germany produce the maximum amount of electricity. The Hungary produce the lowest amount of power.")
# #cutting the whole dateframe based on year 2020
# df_production_2020 = df_production[(df_production.index.year > 2019) & (df_production.index.year < 2021)]
# # which site produce the max solar in year 2020
st.markdown("### 2.Total production in 2020 by different regions")
# df_production_2020.groupby(df_production_2020.index.year).sum().plot(kind="bar",xlabel='Year',ylabel='Energy in Watts',title="Total production in 2020 by different regions", figsize=(16,6))
# st.pyplot(plt.gcf())
st.image("two.jpg")
st.write("The total production of solar electricity by different regions in 2020 is displayed through this graph.")
# df_production_2020.drop('NL_solar_generation_actual', inplace=True, axis=1)
# solar_generation_cols =solar_generation_cols[0:-1]
# #distribution of solar generation
# fig = plt.figure(figsize =(10, 7))
st.markdown("### 3.Distribution of Solar Power in Europe")
# plt.pie(df_production_2020.sum(), labels = solar_generation_cols,autopct='%1.0f%%', radius=1.8 ) 
# # show plot
# plt.show()
# st.pyplot(plt)
st.image("three.jpg")
st.write("The Total production of Solar electricity by different regions in 2020 is displayed through this graph.")
# mean of price per month over 4 years
st.markdown("### 4.The variation of price over different months")
# price_cols = [col for col in df.columns if 'price' in col]
# df_price = df[price_cols]   
# df_price.groupby(df_price.index.month).mean().plot(kind="bar",xlabel='Month',ylabel='price',title="The variation of price over different months", figsize=(16,6))
# st.pyplot(plt.gcf())
st.image("four.jpg")
st.write("The plot exhibit the average Energy price over different months. The month september has highest Energy price among the other months.The Energy price of month May is the lowest.")
#production per month in AT alone
#["AT_solar_generation_actual"]
st.markdown("### 5.Average Monthly solar production of Austria")
# df_production["AT_solar_generation_actual"].groupby(df_production["AT_solar_generation_actual"].index.month).mean().plot(kind="bar",xlabel='Month',ylabel='energy in MWh',title="Average Monthly solar production of Austria", color= "green", figsize = (16,6))
# st.pyplot(plt.gcf())
st.image("five.jpg")
st.write("This plot manifest the amount of solar Energy production over different months in Austria. The month June harvest the maximum amount of solar power and January has the lowest production.")
# taking AT to consideration
# AT_data = [col for col in df.columns if 'AT' in col]
# AT_generation = [col for col in AT_data  if 'generation' in col]
# df_AT_data = df[AT_generation]
#wind over solar 
st.markdown("### 6.Production of wind and solar power in Austria over six years")
# fig, ax = plt.subplots()
# ax.plot(df_AT_data.index, df_AT_data['AT_wind_onshore_generation_actual'])
# ax.plot(df_AT_data.index, df_AT_data['AT_solar_generation_actual'])
# ax.legend(['AT_wind_onshore_generation_actual',"AT_solar_generation_actual"])
# ax.set_title("Production of wind and solar power in Austria over six years")
# ax.set_ylabel("Energy in KWh")
# ax.set_xlabel("Year of Production")
# plt.show()
# st.pyplot(plt)
st.image("six.jpg")
st.write("This plot displays the production of both solar and wind Energy in Austria over 6 years. Austria have more wind Energy production than solar.")