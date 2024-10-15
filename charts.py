import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rand = np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()
ax.hist(rand, bins=15)
st.pyplot(fig)

df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.line_chart(df)

df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.bar_chart(df)

df = pd.DataFrame(np.random.randn(10, 2), columns=['x', 'y'])
st.area_chart(df)

df = pd.DataFrame(
    np.random.randn(500, 2) / [50, 50] + [20.6, 78.9], columns=['lat', 'lon']
)

st.write(df)
st.map(df)
