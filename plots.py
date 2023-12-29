"""
Created on Thu Dec 28 21:15:31 2023

@author: Ryan Milgrim, CFA
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import utility


# %% Main Plot Functions


def PlotAllAssetsPdf(returns, weights, regime_means, regime_covs, colors, n=1000):
    """
    Plots the probability density functions for all assets in the returns DataFrame.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing return data for multiple assets.
    - weights (pd.Series): Weights of Gaussian mixture components, indexed by regime names.
    - regime_means (pd.DataFrame): DataFrame of means for each regime.
    - regime_covs (dict): Dictionary of covariance matrices for each regime.
    - colors (dict): Dictionary of colors for each regime.
    - n (int): Number of points for the PDF plot. Default is 1000.

    Creates a grid of subplots with one subplot per asset, displaying the PDF for each.
    """
    num_assets = len(returns.columns)
    num_rows = int(np.ceil(np.sqrt(num_assets)))
    num_cols = int(np.ceil(num_assets / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))

    for i, asset in enumerate(returns.columns):
        ax = axs[i // num_cols, i % num_cols] if num_assets > 1 else axs
        PlotPdf(ax, asset, weights, returns, regime_means, regime_covs, colors, n)

        # Hide axes for empty subplots if the number of assets is not a perfect square
    for j in range(num_assets, num_rows * num_cols):
       fig.delaxes(axs.flatten()[j])  # Delete any additional axes

    fig.tight_layout()
    plt.show()


# %% Ax Plot Functions


def PlotPdf(ax, asset, weights, returns, regime_means, regime_covs, colors, n=1000):
    """
    Visualizes the distribution of an asset's returns using a histogram and Gaussian Mixture Model (GMM).

    This function plots a histogram of the asset's returns and overlays the probability density functions
    (PDFs) of each identified market regime. The regimes are modeled as components of a Gaussian Mixture Model,
    with each component's contribution weighted by its respective weight in the mixture.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    - asset (str): Asset name for which the PDF is plotted.
    - weights (pd.Series): Weights of the regimes in the GMM, indexed by regime names.
    - returns (pd.DataFrame): DataFrame containing return data for the asset.
    - regime_means (pd.DataFrame): Mean returns for each regime in the GMM.
    - regime_covs (dict): Covariance matrices for each regime in the GMM.
    - colors (dict): Color mapping for each regime.
    - n (int): Number of points to use in plotting the PDFs. Default is 1000.

    The function plots the empirical distribution of returns as a histogram and overlays the PDFs
    of each regime's Gaussian component, along with the combined weighted mixture as a red dashed line.
    """
    bins = int(0.05 * len(returns))
    regime_names = weights.index

    # Plot the historical histogram
    ax.hist(returns[asset], bins=bins, density=True, alpha=0.8, color='grey', label='Historical Histogram')

    # Initialize array to accumulate weighted PDFs
    total_mixture_pdf = np.zeros(n)

    # Plot each Gaussian component of the mixture
    x = np.linspace(min(returns[asset]), max(returns[asset]), n)
    for regime in regime_names:
        mu = regime_means[regime][asset]
        sigma = np.sqrt(regime_covs[regime].loc[asset, asset])
        component_pdf = norm.pdf(x, mu, sigma)
        ax.plot(x, component_pdf, label=f'{regime} ({weights[regime]:.0%})', color=colors[regime], linewidth=2)
        
        # Add the weighted component to the total mixture
        total_mixture_pdf += weights[regime] * component_pdf

    # Plot the combined Gaussian Mixture
    ax.plot(x, total_mixture_pdf, color='red', linewidth=2, linestyle='--', label='Gaussian Mixture')

    # Formatting the plot
    ax.set_yticklabels([])
    SetAxisLimitsBasedOnQuantiles(ax, 'x', returns[asset], formatAsPercent=True)
    ax.legend(loc='upper right', fontsize='x-small', title=f'Density Analysis: {asset}')
    ax.grid(True, linestyle='--', alpha=0.8)
    ReformatLegendLabels(ax)


def plotRegimeViolins(ax, returns, asset, regimes, colors):
    """
    Plots the returns distribution for a specified asset across different market regimes using a violin plot,
    including the full historical data as a reference. The mean of each distribution is marked with a point.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object where the violin plot will be plotted.
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - asset (str): Name of the asset to be analyzed.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict): Dictionary mapping regime names to color values.
    """
    
    asset_returns = returns[asset]
    unique_regimes = sorted(set(regimes))  # Ensuring unique and sorted regime names

    # Prepare data for plotting
    dataset = [asset_returns[regimes == regime].values for regime in unique_regimes]
    dataset.append(asset_returns.values)  # Appending full history data
    
    # Create the violin plot
    parts = ax.violinplot(dataset, showmeans=False, showextrema=False, vert=False)

    # Customize the appearance of the violin plots
    for pc, regime in zip(parts['bodies'], unique_regimes + ['Full History']):
        pc.set_facecolor(colors.get(regime, 'grey'))  # Use 'grey' for "Full History" or missing color
        pc.set_edgecolor('black')
        pc.set_alpha(1)  # Setting alpha value to 1

    # Plotting the mean as a point
    for i, regime in enumerate(unique_regimes + ['Full History']):
        mean_val = np.mean(dataset[i])
        ax.scatter(mean_val, i + 1, color='red', edgecolor='black', s=25)

    # Set the y-axis labels to the regime names
    y_ticks_labels = [f'Regime: {regime}' if not isinstance(regime, str) else regime for regime in unique_regimes]
    y_ticks_labels.append('Full History')  # Label for the full history violin plot
    ax.set_yticks(np.arange(1, len(unique_regimes) + 2))  # +2 to include "Full History"
    ax.set_yticklabels(y_ticks_labels)
    
    # Set limits based on quantiles and percentage format
    SetAxisLimitsBasedOnQuantiles(ax, 'x', asset_returns, (0.0001, 0.9999), True, scale=1.2)

    # Add grid, title, and adjust layout
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'{asset} - Returns by Regime')


def plotRegimeStatsTable(ax, returns, asset, regimes):
    """
    Displays a statistics table for a specified asset's returns across different market regimes.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object where the table will be displayed.
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - asset (str): Name of the asset to be analyzed.
    - regimes (pd.Series): Series indicating the regime for each data point.
    """
    
    # Filter returns for the specified asset
    asset_returns = returns[asset]
    regime_names = sorted(regimes.unique())

    # Calculate statistics for each regime
    regime_stats = pd.DataFrame({regime: utility.CalculateReturnStatistics(asset_returns[regimes == regime]) for regime in regime_names})
    
    # Include full history for comparison
    regime_stats['Full History'] = utility.CalculateReturnStatistics(asset_returns)
    regime_stats = regime_stats.transpose()

    # Formatting the statistics table
    regime_stats['Observations'] = regime_stats['Observations'].astype(int)
    regime_stats['Mean'] = (regime_stats['Mean'] * 100).round(2).astype(str) + '%'
    regime_stats['Vol'] = (regime_stats['Vol'] * 100).round(2).astype(str) + '%'
    regime_stats['Skew'] = regime_stats['Skew'].round(2)
    regime_stats['Kurtosis'] = regime_stats['Kurtosis'].round(2)

    # Add the table to the provided axes
    ax.axis('off')
    table = ax.table(cellText=regime_stats.values,
                     rowLabels=regime_stats.index,
                     colLabels=regime_stats.columns,
                     cellLoc='center', rowLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.85, 2.5)  # Adjust scale as needed

    AddDateDisclaimerToAx(ax, returns, (0.95, 0.05))


# %% Plot Utility Functions


def CleanColors(colorInput='cubehelix', regimes=None):
    """
    Standardizes color input for plotting. Supports direct color mappings (dict) and colormap names (str).

    Parameters:
    - colorInput (str or dict): A Matplotlib colormap name or a dictionary mapping regimes to colors.
    - regimes (pd.Series, optional): Series of regimes. Required if colorInput is a colormap name.

    Returns:
    - dict: A dictionary mapping each regime to its color value.

    Raises:
    - ValueError: If colorInput type is incorrect or if regimes are needed but not provided.

    Examples:
    - Using a colormap name with regimes: CleanColors('viridis', mySeries)
    - Using a direct color mapping: CleanColors({'regime1': 'red', 'regime2': 'blue'})
    """

    if isinstance(colorInput, dict):
        return colorInput

    elif isinstance(colorInput, str):
        if regimes is None:
            raise ValueError("Regimes series must be provided when using a colormap name.")
        
        colormap = plt.get_cmap(colorInput)
        unique_regimes = regimes.unique()
        color_values = [colormap(i) for i in np.linspace(0.15, 0.85, len(unique_regimes))]
        return dict(zip(unique_regimes, color_values))

    else:
        raise ValueError("colorInput must be either a colormap name or a dictionary mapping regimes to colors.")


def SetAxisLimitsBasedOnQuantiles(ax, axis, data, quantileRange=(0.002, 0.998), formatAsPercent=True, scale=1):
    """
    Set the limits of the specified axis based on the quantiles of the data, rounding up to the nearest 0.01.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to which the limits are set.
    - axis (str): Specify 'x' or 'y' to set the corresponding axis limits.
    - data (np.ndarray or pd.Series): Data used to calculate quantile-based limits.
    - quantileRange (tuple): A tuple of two floats representing the lower and upper quantiles.
    - formatAsPercent (bool): If True, formats the axis ticks as percentages.
    - scale (float): Mutiplies the limits to enlarge the axis.

    Raises:
    - ValueError: If the provided axis is neither 'x' nor 'y'.
    - TypeError: If the data is not a numpy array or pandas Series.
    """
    if axis not in ['x', 'y']:
        raise ValueError("Axis must be either 'x' or 'y'.")

    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Data must be either a numpy array or a pandas Series.")

    lowerQuantile, upperQuantile = quantileRange
    lowerLimit, upperLimit = data.quantile([lowerQuantile, upperQuantile])
    
    # Round up to the nearest 0.01
    limit = np.ceil(max(abs(lowerLimit), abs(upperLimit)) * 100 * scale) / 100

    if axis == 'x':
        ax.set_xlim(-limit, limit)
        if formatAsPercent:
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    else:
        ax.set_ylim(-limit, limit)
        if formatAsPercent:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def ReformatLegendLabels(ax):
    """
    Updates the legend labels on an Axes object to include a prefix if the labels are numeric.

    If a label represents an integer, it will be prefixed with "Regime: ".
    If the label is a string, it will be displayed as is.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object with the legend to be updated.
    """
    legend = ax.get_legend()
    if legend:
        new_labels = []
        for text in legend.get_texts():
            label = text.get_text()
            # Check if the label is numeric and prefix with "Regime: " if so
            if isinstance(4, (float, int, np.int32)) or label.isdigit():
                new_label = f"Regime: {label}"
            else:
                new_label = label
            new_labels.append(new_label)
        
        # Set the new labels to the legend
        for text, new_label in zip(legend.get_texts(), new_labels):
            text.set_text(new_label)


def AddDateDisclaimerToAx(ax, time_series, placement=(0.5, -0.1), fontsize='x-small', dateFormat='%B %d, %Y', disclaimerText='As Of'):
    """
    Adds a date disclaimer to a Matplotlib axes (subplot), indicating the latest date of the time series data.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to which the disclaimer is added.
    - time_series (pd.DataFrame or pd.Series): Time series data with a datetime or period index.
    - placement (tuple): Coordinates (x, y) for the disclaimer's location within the axes.
    - fontsize (str): Font size of the disclaimer text (e.g., 'small', 'x-small').
    - dateFormat (str): Format for displaying the date. Default is '%B %d, %Y'.
    - disclaimerText (str): Custom text to precede the date. Default is 'As Of'.

    Raises:
    - ValueError: If 'time_series' does not have a suitable datetime or period index.

    Example:
    - AddDateDisclaimerToAx(ax, time_series, fontsize='small', dateFormat='%Y-%m-%d')
    """

    if isinstance(time_series.index, pd.PeriodIndex):
        last_date = time_series.index.max().end_time
    elif isinstance(time_series.index, pd.DatetimeIndex):
        last_date = time_series.index.max()
    else:
        raise ValueError("The 'time_series' argument must have a datetime or period index.")

    date_disclaimer = last_date.strftime(dateFormat)
    disclaimer_full_text = f'{disclaimerText}: {date_disclaimer}'
    
    ax.text(
        placement[0], placement[1], disclaimer_full_text, 
        horizontalalignment='right', verticalalignment='bottom', 
        fontsize=fontsize, transform=ax.transAxes
    )
