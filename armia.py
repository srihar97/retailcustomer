#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import streamlit as st 
import numpy as np


# In[9]:


get_ipython().system('pip install pymysql')


# In[7]:


from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# In[8]:


def main():
    

    st.title("Drug Forecasting")
    st.sidebar.title("Drug Forecasting")

    # st.radio('Type of Drug')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile,  index_col=0)
        except:
                try:
                    data = pd.read_excel(uploadedFile,  index_col=0)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "root")
    pw = st.sidebar.text_input("password", "Redchief@123")
    db = st.sidebar.text_input("database", "optimization")
    
    
    if st.button("Predict"):
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "Redchief@123",  # passwrd
                               db = "optimization")) # database
        
        
        ###############################################
        st.subheader(":red[Forecast for Test data]", anchor=None)
         
        forecast_test = pd.DataFrame(model.predict(start = data.index[0], end = data.index[-1]))
        results = pd.concat([data,forecast_test], axis=1)
        results.to_sql('forecast_results', con = engine, if_exists = 'replace', index = False, chunksize = 1000)
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(results.style.background_gradient(cmap=cm).set_precision(2))
        
        ###############################################
        st.text("")
        st.subheader(":red[plot forecasts against actual outcomes]", anchor=None)
        #plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(data.forecast_best)
        ax.plot(forecast_test, color = 'red')
        st.pyplot(fig)
        
        ###############################################
        st.text("")
        st.subheader(":red[Forecast for the nest 12 months]", anchor=None)
        
        forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))
        st.table(forecast.style.background_gradient(cmap=cm).set_precision(2))
        
        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


# In[ ]:




