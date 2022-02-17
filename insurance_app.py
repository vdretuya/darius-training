import numpy as np
import pandas as pd
import os
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set (style='white', font_scale=5)

@st.cache
def load_data():
    return pd.read_csv('insurance_regression.csv')

data = load_data()

###Set title
st.title("Insurance Pricing App")
st.write("""From the data below, we built a machine learning-based pricing model to get quotation for each client based on their demographics""")

###Set Sidebar Options
st.sidebar.title("Inurance Pricing App Parameters")
st.sidebar.write("Tweak to change predictions")

#Age
age  = st.sidebar.slider("Age",100, 40, 12)

#BMI
bmi  = st.sidebar.slider("BMI",15, 40, 29)

#Number of children
num_children  = st.sidebar.slider("Number of children",0, 12, 1)

#Gender
gender  = st.sidebar.radio("Gender",('female', 'male'))

if gender == 'male':
    is_female = 0
else:
        is_female = 1
        
#Is Smoker
smoker  = st.sidebar.radio("Smoker?", ('yes', 'no')) 

if smoker == 'yes':
    is_smoker = 1
else:
        is_smoker = 0
        
#Region
region  = st.sidebar.selectbox("Region", ['northwest', 'northeast', 'southeast', 'southwest']) 

if region == 'northeast':
    loc_list = [1,0,0,0]
elif region == 'northwest':
    loc_list = [0,1,0,0]
elif region == 'southeast':
    loc_list = [0,0,1,0]
elif region == 'southwest':
    loc_list = [0,0,0,1]
    
#Price Output
st.subheader('Output Insurance Price')

#Model filename
filename = 'insurance_regression.sav'

#load the model from disk
loaded_model = joblib.load(filename)

#Age, BMI, Number of Childeren, is_female, is_smoker, is_from_NorthEast
prediction = round(loaded_model.predict([[age, bmi, num_children, is_female, is_smoker] + loc_list])[0])

st.write(f"Suggested Insurance Price is: {prediction}")




    




        