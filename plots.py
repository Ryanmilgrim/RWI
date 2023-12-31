__doc__ = """
Created on Sat Dec 16 01:57:12 2023

@author: Ryan Milgrim, CFA

Plots Module for Financial Data Visualization

- plotAllAssetsPdf:
    Generates a grid of subplots, each displaying the probability density function (PDF)
    for an individual asset in the 'returns' DataFrame. It visualizes the return distributions by regime
    for all assets using a Gaussian Mixture Model.

- plotAssetAnalysis:
    Creates a comprehensive plot for each specified asset. This function combines a
    violin plot, a statistics table, and a QQ plot, providing an in-depth analysis of each asset's
    performance in within Regimes.

- plotRegimeMatrix:
    Constructs an NxN correlation matrix plot for asset comparisons under different
    market Regimes. Each off-diagonal cell represents a cluster plot for a pair of assets, and the
    diagonal cells show the PDF for each individual asset.

- plotAllRegimesAssetViolins:
    Displays violin plots for all assets within each market regime. This function
    creates a series of subplots, each dedicated to a specific regime, showcasing the distribution of
    returns for all assets in that regime.

These functions are designed for visualizing the analysis of the main file. 
Please call help() on individual functions for parameter details.
"""


import numpy as np
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


import utility


# %% Main Plot Functions


def plotAllAssetsPdf(returns, weights, regime_means, regime_covs, colors=None, n=1000):
    """
    Plots the probability density functions for all assets in the returns DataFrame.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing return data for multiple assets.
    - weights (pd.Series): Weights of Gaussian mixture components, indexed by regime names.
    - regime_means (pd.DataFrame): DataFrame of means for each regime.
    - regime_covs (dict): Dictionary of covariance matrices for each regime.
    - colors (dict or str): Dictionary of colors for each regime or a color map string.
    - n (int): Number of points for the PDF plot. Default is 1000.

    Creates a grid of subplots with one subplot per asset, displaying the PDF for each.
    """
    num_assets = len(returns.columns)
    num_rows = int(np.ceil(np.sqrt(num_assets)))
    num_cols = int(np.ceil(num_assets / num_rows))
    colors = cleanColors(colors, weights.index)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))

    for i, asset in enumerate(returns.columns):
        ax = axs[i // num_cols, i % num_cols] if num_assets > 1 else axs
        plotPdf(ax, asset, weights, returns, regime_means, regime_covs, colors, n)

        # Hide axes for empty subplots if the number of assets is not a perfect square
    for j in range(num_assets, num_rows * num_cols):
       fig.delaxes(axs.flatten()[j])  # Delete any additional axes

    fig.tight_layout()
    plt.show()


def plotAssetAnalysis(returns, weights, regime_means, regime_covs, regimes, colors=None):
    """
    Creates a combined plot for each asset including a violin plot, a stats table, and a PDF plot.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - weights (pd.Series): Weights of the regimes in the GMM, indexed by regime names.
    - regime_means (pd.DataFrame): Mean returns for each regime in the GMM.
    - regime_covs (dict): Covariance matrices for each regime in the GMM.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict or str): Dictionary of colors for each regime or a color map string.
    """
    colors = cleanColors(colors, weights.index)
    for asset in returns.columns:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax_violin = fig.add_subplot(gs[0, 0])
        ax_table = fig.add_subplot(gs[1, 0])
        ax_qq = fig.add_subplot(gs[:, 1])

        # Plot the violin plot
        plotRegimeViolins(ax_violin, returns, asset, regimes, colors)

        # Plot the stats table
        plotRegimeStatsTable(ax_table, returns, asset, regimes)

        # Plot the PDF plot
        plotRegimeQQPlots(ax_qq, returns, asset, regimes, colors)

        addDateDisclaimerToAx(ax_qq, returns,placement=(0.98, -0.08))
        
        plt.tight_layout()
        plt.suptitle(f'{asset} - Returns by Regime', fontsize=20, y=1.03)

        plt.show()


def plotRegimeMatrix(returns, regimes, weights, regime_means, regime_covs, colors=None, assets=None):
    """
    Creates an NxB correlation matrix by regime. Each cell in the grid represents a cluster plot for a pair of assets,
    and the diagonal cells show the PDF plot for each asset.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - weights (pd.Series): Weights of the regimes in the GMM, indexed by regime names.
    - regime_means (pd.DataFrame): Mean returns for each regime in the GMM.
    - regime_covs (dict): Covariance matrices for each regime in the GMM.
    - colors (dict or str): Dictionary of colors for each regime or a color map string.
    - assets (list, optional): List of asset names to be analyzed. If None, use all columns in returns.
    """
    colors = cleanColors(colors, weights.index)
    if assets is None:
        assets = returns.columns
    n = len(assets)
    fig, axs = plt.subplots(n, n, figsize=(20, 20))

    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if i == j:
                # Diagonal - Plot PDF
                plotPdf(ax, assets[i], weights, returns, regime_means, regime_covs, colors, n=1000, show_labels=False)
            else:
                # Off-diagonal - Plot regime clusters
                plotRegimeClusters(ax, returns, regimes, regime_means, regime_covs, [assets[i], assets[j]], colors, show_labels=False)

            if j == 0:
                ax.set_ylabel(assets[i], fontsize=30)
            if i == 0:
                ax.set_title(assets[j], fontsize=30)

    # Common settings for all subplots
    # for ax in axs.flat:
    #     ax.label_outer()  # Hide x and y labels not on the edge

    # Set a super title for the figure
    fig.suptitle('Regime Correlation Matrix', fontsize=48, y=1.002)

    # Creating a single legend for the entire figure
    handles = [Line2D([0], [0], markersize=15, color=colors[regime], marker='o', linestyle='', label=f'Regime: {regime}') for regime in np.unique(regimes)]
    fig.legend(handles=handles, loc='lower center', ncol=len(np.unique(regimes)), bbox_to_anchor=(0.5, -0.05), fontsize=30)

    plt.tight_layout()
    plt.show()


def plotAllRegimesAssetViolins(returns, regimes, colors, figsize=(12, 8)):
    """
    Creates a plot for each regime, displaying violin plots for all assets in that regime.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict): Dictionary mapping regime names to color values.
    - figsize (tuple): Figure size. Default is (12, 8).
    """

    unique_regimes = regimes.unique()
    num_regimes = len(unique_regimes)
    fig, axs = plt.subplots(1, num_regimes, figsize=figsize)

    if num_regimes == 1:
        axs = [axs]

    for ax, regime in zip(axs, unique_regimes):
        plotAssetViolins(ax, returns, regime, regimes, colors)
        title = f'Regime: {regime}' if not isinstance(regime, str) else regime
        ax.set_title(title, fontsize=16)

    plt.suptitle('Asset Returns by Regime', fontsize=20)
    legend_elements = [Line2D([0], [0], color=color, lw=8, label=f'Regime: {regime}') for regime, color in colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), bbox_to_anchor=(0.5, -0.075), fontsize=16)
    fig.tight_layout()
    plt.show()


# %% Ax Plot Functions


def plotPdf(ax, asset, weights, returns, regime_means, regime_covs, colors, n=1000, show_labels=True):
    """
    Visualizes the distribution of an asset's returns using a histogram and Gaussian Mixture Model (GMM).

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    - asset (str): Asset name for which the PDF is plotted.
    - weights (pd.Series): Weights of the regimes in the GMM, indexed by regime names.
    - returns (pd.DataFrame): DataFrame containing return data for the asset.
    - regime_means (pd.DataFrame): Mean returns for each regime in the GMM.
    - regime_covs (dict): Covariance matrices for each regime in the GMM.
    - colors (dict): Color mapping for each regime.
    - n (int): Number of points to use in plotting the PDFs. Default is 1000.
    - show_labels (bool): If True, show the axis labels, legend, and title. Default is True.
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
        component_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, component_pdf, color=colors[regime], linewidth=2)
        
        # Add the weighted component to the total mixture
        total_mixture_pdf += weights[regime] * component_pdf

    # Plot the combined Gaussian Mixture
    ax.plot(x, total_mixture_pdf, color='red', linewidth=2, linestyle='--')
    setAxisLimitsBasedOnQuantiles(ax, 'x', returns[asset], formatAsPercent=True)

    if show_labels:
        # Add labels and legend if show_labels is True
        for regime in regime_names:
            ax.plot([], [], label=f'{regime} ({weights[regime]:.0%})', color=colors[regime])
        ax.set_title(f'Density Analysis: {asset}')
        ax.legend(loc='upper right', fontsize='small')
        reformatLegendLabels(ax)
    else:
        # Remove axis labels and legend if show_labels is False
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Common formatting
    ax.grid(True, linestyle='--', alpha=0.8)



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
    setAxisLimitsBasedOnQuantiles(ax, 'x', asset_returns, (0.0001, 0.9999), True, scale=1.2)

    # Add grid, title, and adjust layout
    ax.grid(True, linestyle='--', alpha=0.7)


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
    stats = pd.DataFrame({regime: utility.CalculateReturnStatistics(asset_returns[regimes == regime]) for regime in regime_names})
    
    # Include full history for comparison
    stats['Full History'] = utility.CalculateReturnStatistics(asset_returns)
    stats = stats.transpose()[::-1]

    # Formatting the statistics table
    stats['Observations'] = stats['Observations'].astype(int)
    stats['Mean'] = (stats['Mean'] * 100).round(2).astype(str) + '%'
    stats['Vol'] = (stats['Vol'] * 100).round(2).astype(str) + '%'
    stats['Skew'] = stats['Skew'].round(2)
    stats['Kurtosis'] = stats['Kurtosis'].round(2)

    # Add the table to the provided axes
    ax.axis('off')
    stats.index = [f'Regime: {regime}' if not isinstance(regime, str) else regime for regime in stats.index]
    table = ax.table(cellText=stats.values,
                     rowLabels=stats.index,
                     colLabels=stats.columns,
                     cellLoc='center', rowLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.95, 3)


def plotRegimeClusters(ax, returns, regimes, regime_means, regime_covs, assets, colors, show_labels=True):
    """
    Creates a scatter plot with regime clusters overlaid for a pair of assets on a given axis.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object where the plot will be drawn.
    - returns (pd.DataFrame): DataFrame containing return data for assets.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - regime_means (pd.DataFrame): DataFrame of mean vectors for each regime.
    - regime_covs (dict): Dictionary of covariance matrices for each regime.
    - assets (list): List of two asset names to be plotted.
    - colors (dict): Dictionary of colors for each regime.
    - show_labels (bool): If True, show the legend and axis labels. Default is True.
    """
    if len(assets) != 2:
        raise ValueError("Assets parameter must be a list of two asset names.")

    regime_names = np.unique(regimes)

    for regime in regime_names:
        label_data = returns[regimes == regime][assets]
        ax.scatter(label_data[assets[0]], label_data[assets[1]], color=colors[regime], s=2)

        cov_matrix = regime_covs[regime].loc[assets, assets].values
        mean_vector = regime_means[regime][assets].values
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        width, height = 2 * np.sqrt(eigenvalues)

        ax.add_patch(Ellipse(
            xy=mean_vector, 
            width=width, 
            height=height, 
            angle=np.degrees(angle),
            edgecolor=colors[regime], 
            fc='None', lw=2, linestyle='-'
        ))


    setAxisLimitsBasedOnQuantiles(ax, 'x', returns[assets[0]], (0.005, 0.995))
    setAxisLimitsBasedOnQuantiles(ax, 'y', returns[assets[1]], (0.005, 0.995))

    if show_labels:
        title = f'Cluster Analysis: {assets[0]} - {assets[1]}'
        labels = {regime: f'Regime: {regime}' if isinstance(regime, np.int64) else regime for regime in regime_names}
        handles = [
            Line2D([0], [0], color=colors[regime], marker='o', linestyle='', label=labels[regime])
            for regime in regime_names
        ]
        ax.legend(handles=handles, loc='lower right', ncol=len(regime_names), title=title, fontsize='small')

        ax.set_xlabel(assets[0], fontsize='small')
        ax.set_ylabel(assets[1], fontsize='small')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.8)


def plotRegimeQQPlots(ax, returns, asset, regimes, colors, n=1000):
    """
    Plots QQ lines for each regime of a specified asset, along with the full history.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    - returns (pd.DataFrame): DataFrame containing returns for multiple assets.
    - asset (str): The specific asset to plot.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict): Dictionary mapping regime names to color values.
    - distribution (str): The distribution to use for comparison. Default is 'norm'.
    - n (int): Number of points for the QQ plot. Default is 1000.
    """
    extreme_x_values = []
    extreme_y_values = []

    # Plot for each regime
    for regime in regimes.unique():
        regime_rets = returns[asset][regimes == regime]
        qq_line = makeQQLine(regime_rets, n)
        ax.scatter(qq_line.index, qq_line.values, label=regime, color=colors[regime], s=5)
        extreme_x_values.extend([qq_line.index.min(), qq_line.index.max()])
        extreme_y_values.extend([qq_line.values.min(), qq_line.values.max()])

    # Plot for full history
    full_hist = makeQQLine(returns[asset], n)
    ax.scatter(full_hist.index, full_hist.values, label='Full History', s=10)
    extreme_x_values.extend([full_hist.index.min(), full_hist.index.max()])
    extreme_y_values.extend([full_hist.values.min(), full_hist.values.max()])

    # Set axis limits based on extreme values
    ax.set_xlim(min(extreme_x_values), max(extreme_x_values))
    ax.set_ylim(min(extreme_y_values), max(extreme_y_values))

    # Adding a 45-degree reference line
    max_limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([-max_limit, max_limit], [-max_limit, max_limit], color='red', linestyle='--')

    # Reformating legend labels
    ax.legend(loc='upper left')
    reformatLegendLabels(ax)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f"QQ Plots for {asset} Across Regimes")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")


def plotAssetViolins(ax, returns, regime, regimes, colors):
    """
    Plots the returns distribution for each asset within a specified regime using violin plots.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object where the violin plot will be plotted.
    - returns (pd.DataFrame): DataFrame containing asset returns.
    - regime (str or int): The specific regime for which the violins are plotted.
    - regimes (pd.Series): Series indicating the regime for each data point.
    - colors (dict): Dictionary mapping regime names to color values.
    """
    
    # Prepare data for plotting
    returns = returns[returns.columns[::-1]]
    dataset = [returns[asset][regimes == regime].dropna().values for asset in returns.columns]

    # Create the violin plot
    parts = ax.violinplot(dataset, showmeans=False, showextrema=False, vert=False)

    # Customize the appearance of the violin plots
    for pc, asset in zip(parts['bodies'], returns.columns):
        pc.set_facecolor(colors.get(regime, 'grey'))
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Plotting the mean as a point
    for i, asset in enumerate(returns.columns):
        mean_val = np.mean(returns[asset][regimes == regime].dropna())
        ax.scatter(mean_val, i + 1, color='red', edgecolor='black', s=25)

    # Set the y-axis labels to the asset names
    ax.set_yticks(np.arange(1, len(returns.columns) + 1))
    ax.set_yticklabels(returns.columns)
    
    # Set limits based on quantiles and percentage format
    data = returns.transpose().abs().max()[regimes == regime]
    setAxisLimitsBasedOnQuantiles(ax, 'x', data, (0, 1), True)

    ax.grid(True, linestyle='--', axis='x', alpha=0.7)
    ax.set_title(f'Violin Plots for: {regime}' if isinstance(regime, str) else f'Violin Plots for Regime: {regime}')


# %% Plot Utility Functions


def makeQQLine(rets,  n=1000):
    """
    Create a qq line relative to a normal distribution. 
    
    Parameters:
     - rets (pd.Series): a Pandas Series of a single asset's historical returns
     - n (int): The number of points to create for the qqplot.

    Returns:
     - qq (pd.Series): A series representing the line to be drawn on a qqplot. 
    """
   
    points = np.linspace(0, 1, len(rets))
    standard_rets = (rets - rets.mean()) / rets.std()
    sample_quantiles = standard_rets.quantile(points)

    theoritical_quantiles = stats.norm.ppf((np.arange(len(rets)) + 0.5) / len(rets))

    sample_quantiles.index = theoritical_quantiles
    return sample_quantiles


def cleanColors(colorInput=None, regimes=None):
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
    - Using a colormap name with regimes: cleanColors('viridis', mySeries)
    - Using a direct color mapping: cleanColors({'regime1': 'red', 'regime2': 'blue'})
    """

    if colorInput is None:
        colorInput = 'cubehelix'

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


def setAxisLimitsBasedOnQuantiles(ax, axis, data, quantileRange=(0.002, 0.998), formatAsPercent=True, scale=1):
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


def reformatLegendLabels(ax):
    """
    Updates the legend labels on an Axes object to include a prefix if the labels are numeric.

    If a label represents an integer, it will be prefixed with "Regime: ".
    If the label is a string, it will be displayed as is.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object with the legend to be updated.
    """
    legend = ax.get_legend()
    for text in legend.get_texts():
        label = text.get_text()
        regime = label.split()[0]
        if isinstance(regime, (float, int, np.int32)) or regime.isdigit():
            text.set_text(f"Regime: {label}")


def addDateDisclaimerToAx(ax, time_series, placement=(0.5, -0.1), fontsize='x-small', dateFormat='%B %d, %Y', disclaimerText='As Of'):
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
    - addDateDisclaimerToAx(ax, time_series, fontsize='small', dateFormat='%Y-%m-%d')
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

# %% Print Docstring

def doc():
    """Helper function to make module doc string more easily accessable."""
    print(__doc__)
