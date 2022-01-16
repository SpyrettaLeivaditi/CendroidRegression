# Imports

from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# Global counter for figure enumeration
i = 0


def prediction_plot_scorer(model, x, y):
    """ Plot predictions
    :param model: the linear regression model
    :param x: features
    :param y: labels(not used)
    :return: I return the beloved 42, just to return the proper data type
    """
    x_feature = x[:, :1]
    x_plot = x_feature.reshape(len(x_feature), 1)
    # Plot outputs
    global i
    i += 1

    plt.scatter(x_plot, y, color='black')
    plt.title(f'Figure {i}')
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.xticks(())
    plt.yticks(())
    plt.plot(x_plot, model.predict(x), color='red', linewidth=3)
    plt.show()
    return 42


# Project parameter
data = pd.read_csv('voice.csv')
columns = [col for col in list(data.columns) if col != 'centroid' and col != 'label']
features = data.loc[:, columns]
labels = data.loc[:, 'centroid']


# Plot of all data correlation matrix
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
plt.show()

# Normalization of the features
minmax = MinMaxScaler()
features = minmax.fit_transform(features)

# Split data into train and test (random split in every run)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# Initialize model
models = [(LinearRegression(), 'Linear Regression'), (MLPRegressor(random_state=1, max_iter=500), 'MLP Regression')]

for mod in models:
    scoring = {'r2_scr': 'r2',
               'NRMSE_scr': 'neg_root_mean_squared_error',
               'max_error_scr': 'max_error',
               'prediction_plot_scorer': prediction_plot_scorer
               }

    # Cross-validation
    scores = cross_validate(mod[0], x_train, y_train, scoring=scoring, return_estimator=False, cv=10)
    print(f'Training and Evaluating {mod[1]}!!')

    # Mean scores calculation
    for k, v in scores.items():
        if k == 'test_conf_mtx':
            continue
        print(f'{k}: {mean(v)}')
