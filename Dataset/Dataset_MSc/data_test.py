import pandas as pd
import json

"""
Just a test file to check how the dataset is working
"""

import matplotlib.pyplot as plt

# Load the CSV data
GFOC = "/home/dschwarz/Documents/MT/Dataset_MSc/GFOC_RDCDFI.csv"
data = pd.read_csv(GFOC)

# Plot |avg B| against time
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['|avg B|'], label='|avg B| vs Time')
plt.xlabel('Time')
plt.ylabel('|avg B|')
plt.title('Plot of |avg B| against Time')
plt.legend()
plt.grid()
plt.show()