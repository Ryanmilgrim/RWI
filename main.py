"""
Created on Fri Dec 15 20:38:54 2023

@author: Ryan Milgrim, CFA
"""

import utility
import numpy as np
import pandas as pd

from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import MNLogit

from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import RobustScaler


from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


# %% Download data and setup variables

years = 20
freq = 'W'
assets = ['SPY', 'IJH', 'IJR', 'LQD', 'HYG', 'SPTL', 'GLD']

macro = utility.getMacro(freq=freq, years=years)
returns = utility.getReturns(assets=assets, years=years, freq=freq)

end = min(max(macro.index), max(returns.index))
start = max(min(macro.index), min(returns.index))

final_return = returns.tail(1)
returns = returns[start:end]
macro = macro[start:end]


# %% Classify history

def classify_market_regimes(returns, n_components=3, n_init=100, outlier_cutoff=0.01):
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


gmm, gmm_regimes, gmm_means, gmm_covs, gmm_weights = classify_market_regimes(returns)


# %% Defining Plot Functions

def plotPdf(asset, weights, returns, regime_means, regime_covs, colors, n=1000):
    """
    Plots the probability density function (PDF) of an asset's return data.
    
    Parameters:
    - asset: Name of the asset.
    - weights: Weights of Gaussian mixture components, indexed by regime names.
    - returns: DataFrame of asset returns.
    - regime_means: DataFrame of means for each regime.
    - regime_covs: Dictionary of covariance matrices for each regime.
    - colors: Series of colors for each regime.
    - bins: Number of bins for the histogram.
    - n: Number of points for the PDF plot.
    """
    bins = int(0.05 * len(returns))
    fig, ax = plt.subplots()
    regime_names = weights.index

    # Plot the histogram with a lighter color and lower alpha
    ax.hist(returns[asset], bins=bins, density=True, label='Historical Histogram')

    # Plot each Gaussian component of the mixture
    x = np.linspace(-0.2, 0.2, n)
    for regime in regime_names:
        mu = regime_means[regime][asset]
        sigma = np.sqrt(regime_covs[regime].loc[asset, asset])
        component_pdf = norm.pdf(x, mu, sigma)

        # Then plot the colored line on top
        ax.plot(x, component_pdf, label=f'{weights[regime]:.0%} Regime: {regime}', color=colors[regime], linewidth=2)

    # Overlay the combined Gaussian Mixture with a red line
    total_mixture_pdf = [
        weights[regime] * norm.pdf(x, regime_means[regime][asset], np.sqrt(regime_covs[regime][asset][asset]))
        for regime in regime_names
    ]
    ax.plot(x,np.sum(total_mixture_pdf, axis=0), color='red', linewidth=2, label='Gaussian Mixture', linestyle='--')

    set_axis_limits(ax, 'x', returns[asset], (0.0025, 0.9975))
    add_date_disclaimer(fig, returns)

    # Add the legend and format it
    ax.legend(loc='upper right', fontsize='small')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    ax.set_title(f'Density Analysis: {asset}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plotRegimeViolins(returns, asset, regimes, colors):
    """
    Plots the returns distribution for a specified asset across different market regimes using violin plots
    and displays a statistics table for each regime.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - asset (str): Name of the asset to be analyzed.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict): Dictionary mapping regime names to color values.
    """

    # Filter returns for the specified asset and get unique regime names
    returns = returns[asset]
    regime_names = sorted(regimes.unique())

    regime_returns = dict()
    regime_stats = pd.DataFrame()
    for regime in regime_names:
        # Collect returns for each regime
        regime_returns[regime] = returns[regimes == regime]

        # Calculate statistics for each regime
        regime_stats[regime] = get_stats(returns[regimes == regime])
    
    # Include full history for comparison
    regime_returns['Full History'] = returns
    regime_stats['Full History'] = get_stats(returns)
    regime_names = regime_stats.columns

    # Create violin plot with two subplots: one for violins, one for the table
    fig, (ax_violin, ax_table) = plt.subplots(2, 1)

    # Plotting violin plots
    parts = ax_violin.violinplot(regime_returns.values(), showmeans=True, showextrema=False, vert=False)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linestyle('--')
    
    # Set colors for the violin plots
    for body, regime in zip(parts['bodies'], regime_names):
        color = colors.get(regime)
        if color is not None:
            body.set_color(color)
        body.set_edgecolor('black')
        body.set_alpha(0.8)

    # Formatting the violin plot
    ticklabels = [f'Regime: {regime}' if isinstance(regime, (float, int)) else regime for regime in regime_names]
    ax_violin.set_yticks(np.arange(1, len(regime_names) + 1))
    ax_violin.set_yticklabels(ticklabels)   
    ax_violin.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
    ax_violin.grid(True, linestyle='--', axis='x', alpha=0.7)
    ax_violin.set_title(f'{asset} Returns By Regime')

    # Formatting the statistics table
    regime_stats = regime_stats.transpose()
    regime_stats['Observations'] = regime_stats['Observations'].astype(int)
    regime_stats['Mean'] = (regime_stats['Mean'] * 100).round(2).astype(str) + '%'
    regime_stats['Vol'] = (regime_stats['Vol'] * 100).round(2).astype(str) + '%'
    regime_stats['Skew'] = regime_stats['Skew'].round(2)
    regime_stats['Kurtosis'] = regime_stats['Kurtosis'].round(2)
    regime_stats = regime_stats.transpose()

    # Add a table below the violin plot with colors corresponding to rows
    ax_table.axis('off')
    table = ax_table.table(
        cellText=regime_stats.values,
        rowLabels=regime_stats.index,
        colLabels=ticklabels,
        cellLoc='right',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Set limits and add a date disclaimer to the plot
    set_axis_limits(ax_violin, 'x', returns, (0.0001, 0.9999), scale=1.1)
    add_date_disclaimer(fig, returns, (0.98, -0.15))
    
    # Adjust layout for better presentation
    plt.tight_layout(h_pad=-15)
    plt.show()


def plotRegimeClusters(returns, regimes, regime_means, regime_covs, assets, colors):
    """
    Creates a scatter plot with regime clusters overlayed for a pair of assets.
    
    Parameters:
    - returns: DataFrame containing return data for assets.
    - regimes: Series indicating the regime for each data point.
    - regime_means: DataFrame of mean vectors for each regime.
    - regime_covs: Dictionary of covariance matrices for each regime.
    - assets: List of two asset names to be plotted.
    - colors: Series of colors for each regime.
    """
    if len(assets) != 2:
        raise ValueError("Assets parameter must be a list of two asset names.")

    fig, ax = plt.subplots()
    regime_names = np.unique(regimes)

    # Plot an Ellipsoide over each Regime's individual multivariate normal pdf and its scatterplot.
    for regime in regime_names:
        label_data = returns[regimes == regime][assets]
        ax.scatter(label_data[assets[0]], label_data[assets[1]], color=colors[regime], s=2)

        # Calculating the multivariate pdf
        cov_matrix = regime_covs[regime].loc[assets, assets].values
        mean_vector = regime_means[regime][assets].values
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(eigenvalues)

        # Plotting the Ellipse
        ax.add_patch(
            Ellipse(
                xy=mean_vector, 
                width=width, 
                height=height, 
                angle=np.degrees(angle), 
                edgecolor=colors[regime], 
                fc='None', 
                lw=2, 
                linestyle='-'
            )
        )

    # Creating the legend
    ax.legend(
        handles=[Line2D([0], [0], color=colors[regime], label=f'Regime: {regime}') for regime in regime_names],
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=len(regime_names), 
        frameon=True, 
        fontsize='small'
    )

    # Additional plotting
    ax.set_title(f'Cluster Analysis: {assets[0]} - {assets[1]}')
    plt.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel(assets[0], fontsize='small')
    ax.set_ylabel(assets[1], fontsize='small')    
    set_axis_limits(ax, 'x', returns[assets[0]], (0.005, 0.995))
    set_axis_limits(ax, 'y', returns[assets[1]], (0.005, 0.995))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    add_date_disclaimer(fig, returns, (0.98, -0.10))
    plt.show()


def get_stats(returns):
    """
    Calculate basic statistics of a given financial returns series.

    Parameters:
    - returns (pd.Series): A pandas Series representing the asset returns.

    Returns:
    - pd.Series: A pandas Series containing the following statistics:
        - Observations: The number of data points.
        - Mean: The average return.
        - Vol: The standard deviation (volatility) of the returns.
        - Skew: The skewness of the returns.
        - Kurtosis: The excess kurtosis of the returns (Kurtosis - 3).
    """
    return pd.Series({
        'Observations': len(returns),
        'Mean': returns.mean(),
        'Vol': returns.std(),
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis() + 3  # Excess kurtosis
    })



def set_axis_limits(ax, axis, data, quantile_range=(0.002, 0.998), scale=1):
    """
    Set the limits of the specified axis based on the quantiles of the data.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to which the limits are set.
    - axis (str): Specify 'x' or 'y' to set the corresponding axis limits.
    - data (np.ndarray or pd.Series): Data used to calculate quantile-based limits.
    - quantile_range (tuple): A tuple of two floats representing the lower and upper quantiles.
    - scale (float): A scaling factor to adjust the calculated limits.

    Raises:
    - ValueError: If the provided axis is neither 'x' nor 'y'.
    - TypeError: If the data is not a numpy array or pandas Series.
    """
    if axis not in ['x', 'y']:
        raise ValueError("Axis must be either 'x' or 'y'.")

    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Data must be either a numpy array or a pandas Series.")

    lower_quantile, upper_quantile = quantile_range
    lower_limit, upper_limit = data.quantile([lower_quantile, upper_quantile])
    limit = max(abs(lower_limit), abs(upper_limit)) * scale

    if axis == 'x':
        ax.set_xlim(-limit, limit)
    else:
        ax.set_ylim(-limit, limit)



def clean_colors(colors, regimes=None):
    """
    Process and standardize the color input for plotting regimes.

    This function handles two types of color inputs: a dictionary mapping regimes to colors,
    or a string representing a Matplotlib colormap. It returns a standardized pandas Series
    of colors indexed by regime names.

    Parameters:
    - colors (str or dict): A string representing a Matplotlib colormap name or a dictionary mapping regimes to colors.
    - regimes (pd.Series, optional): A pandas Series of regimes. Required if colors is a colormap name.

    Returns:
    - pd.Series: A pandas Series where the index represents regime names and values are color codes.

    Raises:
    - TypeError: If 'colors' is neither a string nor a dictionary.
    - ValueError: If 'colors' is a colormap name but 'regimes' is not provided.
    """
    if isinstance(colors, dict):
        return pd.Series(colors)

    elif isinstance(colors, str):
        if regimes is None:
            raise ValueError("When 'colors' is a colormap name, 'regimes' must be provided.")

        # Updated method for accessing colormaps
        colormap = plt.colormaps[colors]
        unique_regimes = regimes.unique()
        color_values = [rgb2hex(colormap(i)) for i in np.linspace(0.2, 0.8, len(unique_regimes))]
        return pd.Series(color_values, index=unique_regimes)

    else:
        raise TypeError("The 'colors' parameter must be either a dictionary or a colormap name (string).")



def add_date_disclaimer(fig, returns, placement=(0.98, 0.02), fontsize='x-small'):
    """
    Adds a date disclaimer to a matplotlib figure, indicating the latest date of the data.

    The function is particularly useful in financial data visualizations where indicating
    the 'as of' date is essential for contextualizing the analysis.

    Parameters:
    - fig (matplotlib.figure.Figure): The matplotlib figure to which the disclaimer is to be added.
    - returns (pd.DataFrame or pd.Series): A pandas data structure with a datetime index.
    - placement (tuple): Coordinates (x, y) for the location of the disclaimer in the figure space.
                         Values should be in the range [0, 1], with (0, 0) being the bottom left corner.
    - fontsize (str): The font size of the disclaimer text. Acceptable values include standard
                      matplotlib font sizes like 'small', 'medium', 'large', 'x-small', etc.

    Note:
    - The function assumes the latest date in the 'returns' data is the date of interest and formats
      it in a 'Month Day, Year' format for display.
    """
    last_period = pd.Period(returns.index[-1], freq=returns.index.freq)
    date_disclaimer = last_period.end_time.strftime('%B %d, %Y')
    fig.text(
        placement[0], placement[1], f'As Of: {date_disclaimer}', 
        horizontalalignment='right', verticalalignment='bottom', 
        fontsize=fontsize, transform=fig.transFigure
    )


def plotRegimeAnalysis(asset, weights, returns, regimes, regime_means, regime_covs, colors='viridis', plots=0):
    """
    Plots a series of charts for regime analysis of financial assets. Depending on the 'plots' parameter,
    the function generates either density and violin plots for each asset or scatter plots comparing 
    the specified asset against others.

    Parameters:
    - asset (str): The name of the primary asset for comparison in scatter plots.
    - weights (pd.Series): Weights of each regime.
    - returns (pd.DataFrame): DataFrame of asset returns.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - regime_means (pd.DataFrame): Mean returns for each regime.
    - regime_covs (dict): Covariance matrices for each regime.
    - colors (str or dict): A Matplotlib colormap name or a dictionary mapping regimes to colors. Defaults to 'viridis'.
    - plots (int): Determines the type of plots to generate. 
                   0 for PDF and violin plots for each asset, 1 for scatter plots against the specified asset.

    Raises:
    - ValueError: If 'plots' parameter is not 0 or 1.
    """
    colors = clean_colors(colors, regimes)

    if plots == 0:
        for asset in returns.columns:
            plotPdf(asset, weights, returns, regime_means, regime_covs, colors)
            plotRegimeViolins(returns, asset, regimes, colors)
    elif plots == 1:
        for x_asset in returns.drop(columns=asset).columns:
            plotRegimeClusters(returns, regimes, regime_means, regime_covs, [x_asset, asset], colors)
    else:
        raise ValueError("Invalid value for 'plots'. Use 0 for PDF and violin plots, 1 for scatter plots.")




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
    colors = clean_colors(colors, actual_regimes)
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

