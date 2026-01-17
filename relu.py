import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation", layout="wide")
st.title("ReLU Activation Function")

st.write("**ReLU(x) = max(0, x)** — outputs 0 for negative inputs and linear for positive inputs.")

# Sidebar controls
st.sidebar.header("Controls")
x_min = st.sidebar.slider("x minimum", -20, -1, -10)
x_max = st.sidebar.slider("x maximum", 1, 20, 10)
n_points = st.sidebar.slider("Number of points", 50, 2000, 400, step=50)

x = np.linspace(x_min, x_max, n_points)
y = np.maximum(0, x)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Values (sample)")
    st.dataframe({"x": x[:10], "ReLU(x)": y[:10]})

with col2:
    st.subheader("Plot")
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("ReLU(x)")
    ax.grid(True)
    st.pyplot(fig)
