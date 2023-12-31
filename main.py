"""
Created on Fri Dec 15 20:38:54 2023

@author: Ryan Milgrim, CFA
"""

import numpy as np
import pandas as pd

from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import MNLogit

from sklearn.mixture import BayesianGaussianMixture
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import RobustScaler


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import utility, plots


# %% Download data and setup variables

years = 20
freq = 'W'

macro = utility.getMacro(freq=freq, years=years)
returns = utility.getReturns(years=years, freq=freq)
macro, returns = utility.alignAndShiftDataFrames(macro, returns)



# %% Classify history

def classify_market_regimes(returns, n_components=3, n_init=100, outlier_cutoff=0.002):
    """
    Classifies historical asset returns into market regimes using Bayesian Gaussian Mixture Models.

    The function fits an Empirical Covariance model to identify and exclude outliers before fitting
    a Bayesian Gaussian Mixture Model to the filtered returns. It returns the fitted model along
    with the identified regimes, their means, covariance matrices, and the model's component weights.

    Parameters:
    - returns (pd.DataFrame): A DataFrame of historical return data for assets.
    - n_components (int): The number of mixture components (regimes) to identify. Default is 3.
    - n_init (int): The number of initializations to perform for the clustering algorithm. Default is 100.
    - outlier_cutoff (float): The percentage of the largest Mahalanobis distances to consider as outliers. Default is 0.015.

    Returns:
    - gmm (BayesianGaussianMixture): The fitted Bayesian Gaussian Mixture model.
    - regimes (pd.Series): A Series mapping each time period to a regime.
    - means (pd.DataFrame): A DataFrame where each column represents the mean returns of a regime.
    - covs (dict of pd.DataFrame): A dictionary mapping each regime to its covariance matrix.
    - weights (pd.Series): A Series representing the weight of each regime in the mixture model.

    Example:
    gmm, regimes, means, covs, weights = classify_market_regimes(returns)
    """

    # Fit an Empirical Covariance model to find outliers
    cov_model = EmpiricalCovariance().fit(returns)
    outlier_scores = pd.Series(cov_model.mahalanobis(returns), index=returns.index)
    outlier_threshold = int(len(returns) * outlier_cutoff)
    outliers = outlier_scores.nlargest(outlier_threshold).index
    filtered_returns = returns.drop(outliers)

    # Fit a Bayesian Gaussian Mixture Model to the filtered data
    gmm = BayesianGaussianMixture(n_components=n_components, n_init=n_init, init_params='random_from_data')
    gmm.fit(filtered_returns)

    # Classify each period's returns into regimes
    regimes = pd.Series(gmm.predict(returns), index=returns.index, name='Regimes')

    # Extract the regime means and covariance matrices
    means = pd.DataFrame({regime: gmm.means_[regime] for regime in range(n_components)}, index=returns.columns)
    covs = {regime: pd.DataFrame(gmm.covariances_[regime], index=returns.columns, columns=returns.columns)
            for regime in range(n_components)}

    # Record the weight of each regime in the model
    weights = pd.Series(gmm.weights_, index=range(n_components), name='Weights')

    return gmm, regimes, means, covs, weights


gmm, regimes, regime_means, regime_covs, weights = classify_market_regimes(returns)


# %% Plot Function


plotRegimeAnalysis(
    asset='SPY',
    weights=gmm_weights,
    returns=returns,
    regimes=gmm_regimes,
    regime_means=gmm_means,
    regime_covs=gmm_covs,
    colors='twilight',
    plots=0
    )


plotRegimeAnalysis(
    asset='SPY',
    weights=gmm_weights,
    returns=returns,
    regimes=gmm_regimes,
    regime_means=gmm_means,
    regime_covs=gmm_covs,
    colors='twilight',
    plots=1
    )


# %% Fit Logit Model

def predict_market_regime(macro_data, regimes, single_observation=None, model_alpha=1):
    """
    Predicts the current market regime based on macroeconomic data using a regularized multinomial logistic regression model,
    applying robust scaling to the features to mitigate the influence of outliers.

    Parameters:
    - macro_data (pd.DataFrame): A DataFrame containing macroeconomic indicators.
    - regimes (pd.Series): A Series containing the identified historical market regimes.
    - model_alpha (float): The regularization strength; must be a positive float. Larger values specify stronger regularization. Default is 0.1.

    Returns:
    - predictions (pd.DataFrame): A DataFrame containing the predicted probabilities for each regime.
    - pred (int): The index of the predicted market regime.
    - prob (str): The probability of the predicted market regime as a formatted string.

    Raises:
    - ValueError: If the macro_data DataFrame is empty.
    - Exception: If the model fitting process encounters an error.
    """
    if macro_data.empty:
        raise ValueError("The macro_data DataFrame is empty. Prediction requires non-empty macroeconomic data.")

    # Scale the macroeconomic features using RobustScaler
    scaler = RobustScaler()
    scaled_macro_data = scaler.fit_transform(macro_data)
    scaled_macro_data = pd.DataFrame(scaled_macro_data, index=macro_data.index, columns=macro_data.columns)

    # Add a constant term for the intercept
    scaled_macro_with_constant = add_constant(scaled_macro_data)

    # Fit the model
    model = MNLogit(regimes.values, scaled_macro_with_constant.values)
    results = model.fit_regularized(alpha=model_alpha)

    # If single prediction is true
    if single_observation is not None:
        data = np.insert(scaler.transform(single_observation).flatten(), 0, 1)
        return results.predict(data)[0]

    return results.predict(scaled_macro_with_constant)

def generate_out_of_sample_predictions(macro_data, regimes, start_points, model_alpha=1):
    """
    Generates out-of-sample predictions for market regimes by incrementally using past data up to each point.

    Parameters:
    - macro_data (pd.DataFrame): DataFrame containing macroeconomic indicators.
    - regimes (pd.Series): Series containing the identified historical market regimes.
    - start_points (int): The number of initial data points to start with.
    - model_alpha (float): Regularization strength for the model. Default is 1.

    Returns:
    - out_of_sample_predictions (pd.DataFrame): DataFrame containing out-of-sample predicted probabilities.
    """
    if start_points >= len(macro_data):
        raise ValueError("start_points should be less than the length of macro_data.")

    predictions = dict()
    for i in range(start_points, 1 + len(macro_data)):
        observed_macro_data = macro_data.head(i)
        observed_regimes = regimes.head(i)[1:]
        unobserved_macro_data = observed_macro_data.tail(1)
        observed_macro_data = observed_macro_data[:-1]

        # Make a single prediction using the predict_market_regime function
        pred = predict_market_regime(observed_macro_data, observed_regimes, unobserved_macro_data)
        predictions[unobserved_macro_data.index[0] + 1] = pred

    predictions = pd.DataFrame(predictions).transpose()
    return predictions

# Example usage:
# Assuming 'macro_data' is your macroeconomic indicators DataFrame,
# 'regimes' is your market regimes Series, and you want to start with 100 data points.
predictions = generate_out_of_sample_predictions(macro, gmm_regimes, 515)

# %%


def plot_predicted_probabilities(predictions, actual_regimes, colors):
    """
    Plots predicted regime probabilities as a time series stackplot, including an actual regime color bar. 

    This function visualizes the probabilistic forecasts of market regimes over time. The plot includes a stackplot showing the predicted probabilities for each regime, with each regime color-coded for clarity. The subplot's title displays the most recent prediction with its probability. The function also formats the x-axis with month and year labels and adds a custom date disclaimer for contextual information.

    Parameters:
    - predictions (pd.DataFrame): A DataFrame containing the predicted probabilities for each regime over time.
    - actual_regimes (pd.Series): A Series containing the actual regime names, used for coloring the plot.
    - colors (dict): A dictionary mapping regime names to color values for the plot.
    """
    fig, ax = plt.subplots()

    # Plotting predicted probabilities
    dates = predictions.index.end_time.date
    colors = CleanColors(colors, actual_regimes)
    ax.stackplot(
        dates,
        predictions.transpose().values,
        colors=colors,
        labels=[f'Regime: {regime}' for regime in predictions.columns]
    )

    prob = predictions.iloc[-1].max()
    pred = predictions.iloc[-1].idxmax()
    pred = f'Regime: {pred}' if isinstance(pred, (float,  int)) else pred
    
    plt.suptitle('Regime Forecast - Latest Prediction')
    ax.set_title(f'{pred} with {round(prob * 100, 1)}% probability', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim(min(dates), max(dates)), ax.set_ylim(0, 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b, %Y'))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

                                 

    # Adjust legend location
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=len(predictions.columns), fontsize='small')
    add_date_disclaimer(fig, returns, (0.98, -0.13))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Example usage
plot_predicted_probabilities(predictions, gmm_regimes, 'twilight')





# %%

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

def plot_multiclass_roc(predictions, actual_regimes):
    """
    Plots the ROC curve for each class in a multiclass classification, considering shifted regimes.

    Parameters:
    - predictions (pd.DataFrame): DataFrame containing the predicted probabilities for each regime.
    - actual_regimes (pd.Series): Series containing the actual shifted regimes.
    """
    # Shift the actual regimes to align with predictions
    shifted_regimes = actual_regimes.shift(-1).dropna()
    aligned_predictions = predictions.loc[shifted_regimes.index]

    # Prepare data for ROC analysis
    n_classes = predictions.shape[1]
    actual_binarized = label_binarize(shifted_regimes, classes=range(n_classes))

    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual_binarized[:, i], aligned_predictions.iloc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC with Shifted Regimes')
    plt.legend(loc="lower right")
    plt.show()



plot_multiclass_roc(predictions, gmm_regimes)

