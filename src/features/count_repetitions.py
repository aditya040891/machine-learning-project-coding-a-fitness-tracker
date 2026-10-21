import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2


# --------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------

df = pd.read_pickle('../../data/interim/01_data_processed.pkl')

df = df[df['label'] != 'rest']

acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2
df['acc_r'] = np.sqrt(acc_r)
df['gyr_r'] = np.sqrt(gyr_r)


# -------------------------------------------------------------------------
# Split data
# -------------------------------------------------------------------------

bench_df = df[df['label'] == 'bench']
squat_df = df[df['label'] == 'squat']
row_df = df[df['label'] == 'row']
ohp_df = df[df['label'] == 'ohp']
dead_df = df[df['label'] == 'dead']


# -------------------------------------------------------------------------
# Visualize data to identify patterns
# -------------------------------------------------------------------------

plot_df = bench_df.copy()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_x'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_y'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_z'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_r'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_x'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_y'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_z'].plot()
plt.show()
plt.close()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_r'].plot()
plt.show()
plt.close()


# ------------------------------------------------------------------------
# Configure LowPassFilter
# ------------------------------------------------------------------------














