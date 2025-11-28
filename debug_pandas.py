import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Sys Path: {sys.path}")

try:
    import numpy
    print(f"Numpy file: {numpy.__file__}")
    print(f"Numpy version: {numpy.__version__}")
except ImportError as e:
    print(f"Numpy Import Error: {e}")

try:
    import pandas
    print(f"Pandas file: {pandas.__file__}")
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Pandas Import Error: {e}")
except Exception as e:
    print(f"Pandas Error: {e}")
