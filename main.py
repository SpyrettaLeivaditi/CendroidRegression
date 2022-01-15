# Imports

from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler

# Global counter for figure enumeration
i = 0


def confusion_matrix_scorer(model, x, y):
    """ Plot the confusion matrix
    :param model: the linear regression model
    :param x: features
    :param y: labels
    :return: I return the beloved 42, just to return the proper data type
    """
    ConfusionMatrixDisplay.from_estimator(model, x, y)
    global i
    i += 1
    plt.title(f'Figure {i}')
    plt.show()
    return 42


# Project parameter
data = pd.read_csv('voice.csv')
columns = [col for col in list(data.columns) if col != 'centroid' and col != 'label']
features = data.loc[:, columns]
labels = data.loc[:, 'centroid']


# Plot of all data correlation matrix
plt.figure(figsize=(40, 40))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')

# Normalization of the features
minmax = MinMaxScaler()
features = minmax.fit_transform(features)

# Split data into train and test (random split in every run)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Initialize model
linear_regr = LinearRegression()

scoring = {'r2_scr': 'r2',
           'NRMSE_scr': 'neg_root_mean_squared_error'}


# Cross-validation
scores = cross_validate(linear_regr, x_train, y_train, scoring=scoring, return_estimator=False, cv=10)

# Mean scores calculation
for k, v in scores.items():
    if k == 'test_conf_mtx':
        continue
    print(f'{k}: {mean(v)}')
