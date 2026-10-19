import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------------------

df_train = df.drop(['participant', 'category', 'set'], axis=1)

print(df_train['label'])

X = df_train.drop('label', axis=1).copy()
y = df_train['label'].copy()


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25,
                                                    random_state=42,
                                                    stratify=y)

print(y_train.value_counts())
print(y_test.value_counts())


fig, ax = plt.subplots(figsize=(10,5))
df_train['label'].value_counts().plot(
        kind='bar', ax = ax, color = "lightblue", label="Total"
    )
y_train.value_counts().plot(kind='bar', 
                            ax = ax,
                            color = 'dodgerblue',
                            label='Train')
y_test.value_counts().plot(kind='bar',
                           ax=ax,
                           color='royalblue',
                           label='Test')
plt.legend()
plt.show()
plt.close()


# -------------------------------------------------------------------------
# Split feature subsets
# -------------------------------------------------------------------------

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

square_features = ['acc_r', 'gyr_r']

pca_features = ['pca_1', 'pca_2', 'pca_3']

time_features = [f for f in df_train.columns if '_temp' in f]

frequency_features = [f for f in df_train.columns if '_freq' in f or '_pse' in f]

cluster_features = ['cluster']


print("Basic Features:", len(basic_features))
print("Square Features:", len(square_features))
print("PCA Features:", len(pca_features))
print("Time Features:", len(time_features))
print("Frequency Features:", len(frequency_features))
print("Cluster Features:", len(cluster_features))


feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# --------------------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------------------


