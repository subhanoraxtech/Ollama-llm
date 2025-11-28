import sys
try:
    import numpy
    print(f"Numpy: {numpy.__version__} at {numpy.__file__}")
except ImportError as e:
    print(f"Numpy Import Error: {e}")

try:
    import scipy
    print(f"Scipy: {scipy.__version__} at {scipy.__file__}")
except ImportError as e:
    print(f"Scipy Import Error: {e}")

try:
    import sklearn
    print(f"Sklearn: {sklearn.__version__} at {sklearn.__file__}")
    from sklearn.cluster import KMeans
    print("Sklearn KMeans imported successfully")
except ImportError as e:
    print(f"Sklearn Import Error: {e}")
except Exception as e:
    print(f"Sklearn Error: {e}")
