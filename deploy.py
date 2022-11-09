#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pickle
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


# In[27]:


pickle_in = open(r"C:\Users\SRIHARI KARANAM\Downloads\classifier.pkl",'rb')
model = pickle.load(pickle_in)


# In[28]:


@st.cache()

def prediction(recency, frequency, monetary):
    
    prediction= model.predict(pd.DataFrame([[recency, frequency, monetary]]))
    
    return prediction


# In[29]:


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


# In[30]:


if __name__== '__main__':
    main()


# In[ ]:




