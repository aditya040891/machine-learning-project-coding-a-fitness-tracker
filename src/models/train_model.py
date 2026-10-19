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

learner = ClassificationAlgorithms()

max_features = 10

selected_features, ordered_features, ordered_scores = learner.forward_selection(max_features, 
                                                                                X_train, 
                                                                                y_train)

selected_features = [
 'pca_1',
 'duration',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_y_temp_mean_ws_5',
 'gyr_x_freq_10.714_Hz_ws_14',
 'gyr_y_freq_14.286_Hz_ws_14',
 'acc_y_freq_3.571_Hz_ws_14',
 'acc_x_freq_10.714_Hz_ws_14',
 'gyr_r_freq_10.714_Hz_ws_14',
 'acc_r']


plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()
plt.close()


# ------------------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# ------------------------------------------------------------------------

possible_feature_sets = [
        feature_set_1,
        feature_set_2,
        feature_set_3,
        feature_set_4,
        selected_features
    ]

feature_names = [
        'Feature Set 1',
        'Feature Set 2',
        'Feature Set 3',
        'Feature Set 4',
        'Selected Features'
    ]

iterations = 1

score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])










