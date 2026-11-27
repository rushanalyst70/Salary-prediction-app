import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title="salary prediction",layout="centered")
model=joblib.load("salary_training_model.pkl")
st.title("salary prediction App")
st.write("Predict salary based on years of professional experience")
st.sidebar.header("Employee information")
exp=st.sidebar.number_input("years of experience",min_value=20,max_value=60,step=1)
if st.button("predict salary"):
    input=np.array([[exp]])
    pred=model.predict(input)
    #to display function
    st.success(f"predicted salary={pred}")
