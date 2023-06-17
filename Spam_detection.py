# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:23:26 2023

@author: shaik
"""
import numpy as np
import pandas as pd
import pickle
import streamlit as st

def pred(input_data):
    fName = "C:\\Users\\shaik\\Downloads\\trained.sav"
    fName2 = "C:\\Users\\shaik\\Downloads\\labels"
    data = pd.read_csv('C:\\Users\\shaik\\Downloads\\spam_mail_data.csv')
    loaded_model = pickle.load(open(fName,"rb"))
    labels = pickle.load(open(fName2,"rb"))
    input_data = labels[0]
    input_data = np.asarray(input_data)
    
    var = labels[0].reshape(1,-1)
    pred = loaded_model.predict(var)
    return pred
    
def Deploy():
    st.title("Email Spam Detection")
    
    Category = st.text_input('Message')
    result = ""
    if st.button("Show Result"):
        input_data = Category
        result = pred(input_data)
    st.success(result)
    
Deploy()
