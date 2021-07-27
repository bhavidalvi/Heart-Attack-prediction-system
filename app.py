import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from pickle import load
from sklearn.neighbors import KNeighborsClassifier


st.title("Heart Prediction System")
model = load(open('knn.sav','rb'))
Age  = st.number_input("Please Input Age")
Sex = st.number_input("Please Input Gender(1 for Male , 0 for Female)")
cp = st.number_input("Please Input cp")
Trtbps = st.number_input("Please Input trtbps")
chol = st.number_input("Please enter Chol")
fbs  = st.number_input("Please enter fbs")
restecg = st.number_input("Please enter Input restecg")
thalachh = st.number_input("Please enter Input thalachh")
exng = st.number_input("Please enter Input exng")
oldpeak = st.number_input("Please enter Input oldpeak")
slp = st.number_input("Please enter Input slp")
caa = st.number_input("Please enter Input caa")
thall = st.number_input("Please enter Input thall")
data = [Age,Sex,cp,Trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]
data=np.array(data).reshape(1,13)
input=pd.DataFrame(data,index = [0],columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall'])
df = pd.read_csv("heart.csv")
df=df.drop("output",axis=1)
input = df.append(input)
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
input = scalerX.fit_transform(input)
input = pd.DataFrame(input).iloc[-1:,]
st.subheader('Predicted Result')
if (st.button('predict')):
    predict = model.predict(input)
    st.write('there is no chance of heart attack' if predict==0 else 'there is chance of heart attack ')

