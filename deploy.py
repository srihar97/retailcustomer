#!/usr/bin/env python
# coding: utf-8

# In[26]:
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("classifier.pkl",'rb')
model = pickle.load(pickle_in)

@st.cache()

def prediction(recency, frequency, monetary):
    
    prediction= model.predict(pd.DataFrame([[recency, frequency, monetary]]))
    
    return prediction


def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Retailer Classification in Pharma </h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    frequency = st.slider('frequency:', 1 , 180)
    monetary = st.slider('monetary:', 30 , 50000)
    recency = st.slider('recency:', 1 , 15)
    result = ""
    
    if st.button('classify'):
        result = prediction(recency, frequency, monetary)
        st.success(f'The retailer belongs to the cluster {result[0]:.0f}')


if __name__== '__main__':
    main()


# In[ ]:




