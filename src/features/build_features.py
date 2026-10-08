import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
import warnings
warnings.simplefilter(action='ignore')
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenets.pkl')

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# ---------------------------------------------------------------------------
# Dealing with missing values (imputation)
# ---------------------------------------------------------------------------

df.info()

df.isna().sum()

subset = df[df["set"] == 44]['gyr_y'].plot()


for col in predictor_columns:
    df[col] = df[col].interpolate()

df.isna().sum()


# -------------------------------------------------------------------------
# Calculate set duration
# -------------------------------------------------------------------------


df[df['set'] == 25]['acc_y'].plot()

df[df['set'] == 50]['acc_y'].plot()

duration = df[df['set'] == 25].index[-1] - df[df['set'] == 25].index[0]

duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    end = df[df['set'] == s].index[-1]
    duration = end - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds
    
df['duration'].unique()

duration_df = df.groupby('category')['duration'].mean()

duration_df[0] / 5  # heavy reps
duration_df[1] / 10 # medium reps         

# -------------------------------------------------------------------------
# Buttersworth LowPass Filter 
# -------------------------------------------------------------------------












