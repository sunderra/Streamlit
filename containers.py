import streamlit as st

st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")
st.sidebar.button("Click Me")
st.sidebar.radio("Pick your gender",["Male","Female"])

st.checkbox('Yes')
st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)
