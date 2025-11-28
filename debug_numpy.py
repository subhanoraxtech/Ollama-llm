import sys
try:
    import numpy
    print(f"Numpy file: {numpy.__file__}")
    print(f"Numpy version: {numpy.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
print(f"Sys path: {sys.path}")
