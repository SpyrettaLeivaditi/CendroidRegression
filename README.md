## Predicting Voice Centroid Frequency, using Linear Regression and a Multi Layer Perceptron 

### Description 

The code in this repository deals with the problem of predicting the centroid frequency of a voice clip, based on other acoustic characteristics. We formulate the problem as a regression task and we use (and compare) Linear Regression and a Multi Layer Perceptron to perform the regression.

This task is part of an assignment in a Machine Learning (ML) course, and focuses on implementing a regression algorithm for a particular problem, evaluate the performance and compare it with any existing implementations.

### Data 

#### Data Location
The dataset we used in this implementation is freely and openly available in [Kaggle](https://www.kaggle.com/primaryobjects/voicegender). It consists of 3,168 recorded voice samples, collected from male and female speakers, and pre-processed to extract a number of acoustic features.

#### Data columns - Data types

The dataset consists of 21 columns:

- **meanfreq**: mean frequency (in kHz); float
- **sd**: standard deviation of frequency; float
- **median**: median frequency (in kHz); float
- **Q25**: first quantile (in kHz); float
- **Q75**: third quantile (in kHz); float
- **IQR**: interquantile range (in kHz); float
- **skew**: skewness; float
- **kurt**: kurtosis; float
- **sp.ent**: spectral entropy; float
- **sfm**: spectral flatness; float
- **mode**: mode frequency; float
- **centroid**: frequency centroid; float
- **meanfun**: mean fundamental frequency measured across acoustic signal; float
- **minfun**: minimum fundamental frequency measured across acoustic signal; float
- **maxfun**: maximum fundamental frequency measured across acoustic signal; float
- **meandom**: mean of dominant frequency measured across acoustic signal; float
- **mindom**: minimum of dominant frequency measured across acoustic signal; float
- **maxdom**: maximum of dominant frequency measured across acoustic signal; float
- **dfrange**: range of dominant frequency measured across acoustic signal; float
- **modindx**: modulation index; float
- **labels**: male/female (string)

The target column we want to predict is the "centroid"

### Running the code

To run the code first install the required dependencies:

```
pip install -r requirements.txt
```
Then you can run the classification task by running:

```
python main.py
```

The output consists of a set of calculated evaluation metrics printed in the console, and a number of plots, all illustrating the regression's performance.

### What our code does

In our code, we execute the following workflow:

- First we read the data from a csv file and split them into features and labels. The "labels" column (male/female) is completely dropped as it is not an acoustic feature and does not play an important role in our investigation. 
- Then we plot a data correlation matrix, in the form of a heatmap, that shows the pairwise correlation between the different columns of the data.
- Then, we normalize the feature values to the range [0,1]  using a MinMax scaler. This is necessary as the different features have different ranges that may negatively affect the classifier.
- Then, we split the data into train and test sets, with the train set consisting of 80% of the data and the test one of 20%. This splitting is random in every run.
- Then, we initialize a Logistic Regression model and a Multi Layer Perceptron model.
- Then, we define the scoring metrics that we want to evaluate the two models with. These are the R2 (coefficient of determination) regression score function, the mean squared error regression loss, and the maximum residual error. We also define a function prediction_plot_scorer that plots the actual and the predicted values of the centroid with respect to the mean frequency feature.
- Then we run and evaluate the two regression models by performing a 10-fold cross-validation. We use cross-validation to ensure our evaluation is not biased from the random splitting of the data. 
- Finally, we print the average values of the scoring metrics, and plot for each fold the actual and predicted values of the centroid with respect to the mean frequency feature.

### Evaluation Results

The Linear Regression model achieved an R2 score of 1.0, a mean squared error regression loss of -5.16e-17 and a maximum residual error of -1.46e-16. 

The Multi Layer Perceptron, on the other hand, achieved an R2 score of 0.78, a mean squared error regression loss of -0.014 and a maximum residual error of -0.053. 

As such, the Linear Regression model clearly outperformed the Multi Layer Perceptron. This difference is also reflected in the plots; in the case of Linear Regression the predicted centroid values (in red) correlate almost perfectly with the actual ones (in black), while in the case of the Multi Layer Perceptron the predicted values are all around the place.







