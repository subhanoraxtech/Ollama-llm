import streamlit as st
import sys
import os

st.write("## Environment Debugger")
st.write(f"**CWD:** `{os.getcwd()}`")
st.write("**Sys Path:**")
st.write(sys.path)

try:
    import numpy
    st.success(f"Numpy imported: `{numpy.__file__}` version `{numpy.__version__}`")
except Exception as e:
    st.error(f"Numpy failed: {e}")

try:
    import pandas
    st.success(f"Pandas imported: `{pandas.__file__}` version `{pandas.__version__}`")
except Exception as e:
    st.error(f"Pandas failed: {e}")
