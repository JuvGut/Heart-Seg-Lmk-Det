import streamlit as st
import pandas as pd

st.write("""
# Sales Model
         Below are our sales predictions for this customer.
""")

df = pd.read_csv('/home/juval.gutknecht/Projects/Data/results/inference_results_aaa/report/tableau_case_summary.csv')
st.line_chart(df)

st.write("""
# The end
""")