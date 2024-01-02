"""
Created on Sat Dec 16 01:05:31 2023

@author: Ryan Milgrim, CFA
"""

import plots
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.covariance import EmpiricalCovariance


class Stem:
    """
    Base class for various analytical methods in financial data analysis.
    Acts as a foundation for more specific data analysis and modeling classes.
    """

class Classification(Stem):
    """
    Base class for classification strategies in financial data analysis. This class
    is designed to handle preprocessing steps common to various classification
    methodologies, like outlier detection.

    Attributes:
        returns (pd.DataFrame): DataFrame containing historical return data.
        inlier_pct (float): Percentage cutoff for determining inliers based on Mahalanobis distance.

    Methods:
        determineOutliers: Identifies and filters outliers in the return data.
        plotAssetReturns: Visualizes asset returns for each classified regime using violin plots.
    """

    def __init__(self, returns, inlier_pct=0.99):
        self.returns = returns
        self.inlier_pct = inlier_pct

        # Fit and assign classifications
        self.fit(), self.assign()

    def assign(self):
        self.means = None
        self.covs = None
        self.weights = None

    def determineOutliers(self):
        if not isinstance(self.inlier_pct, float):
            raise ValueError(f'Inlier Percent must be a float, not {type(self.inlier_pct)}')
        
        if self.inlier_pct < 0 or self.inlier_pct > 1:
            raise ValueError(f'Inlier Percent must be between 0 and 1, not {self.inlier_pct}')

        cov_model = EmpiricalCovariance().fit(self.returns)
        outlier_scores = pd.Series(cov_model.mahalanobis(self.returns), index=self.returns.index)
        self.inliers = outlier_scores < outlier_scores.quantile(self.inlier_pct)

    def plotPdfs(self, colors=None, n=1000):
        """
        Plots the probability density functions for all assets in the returns DataFrame.

        Parameters:
        - colors (dict or str, optional): Dictionary of colors for each regime or a colormap string.
        - n (int): Number of points for the PDF plot. Default is 1000.
        """
        plots.plotPdfs(self.returns, self.regime_weights, self.means, self.covs, colors, n)

    def plotReturns(self, colors=None, figsize=(12, 8)):
        """
        Creates a plot for each regime, displaying violin plots for all assets in that regime.

        Parameters:
        - colors (dict or str, optional): Dictionary of colors for each regime or a colormap string.
        - figsize (tuple): Figure size. Default is (12, 8).
        """
        plots.plotReturns(self.returns, self.regimes, colors, figsize)

    def plotScorecards(self, colors=None):
        """
        Creates a comprehensive plot for each asset, combining a violin plot, 
        a statistics table, and a PDF plot to provide in-depth analysis.
        
        Parameters:
        - colors (dict or str, optional): Dictionary of colors for each regime or a colormap string.
        """
        plots.plotScorecards(self.returns, self.regime_weights, self.means, self.covs, self.regimes, colors)

    def plotCorr(self, colors=None, assets=None):
        """
        Constructs an NxB correlation matrix by regime, comparing pairs of assets 
        with cluster plots and individual asset distributions on the diagonals.
        
        Parameters:
        - colors (dict or str, optional): Dictionary of colors for each regime or a colormap string.
        - assets (list, optional): List of asset names to be analyzed.
        """
        plots.plotCorr(self.returns, self.regimes, self.regime_weights, self.means, self.covs, colors, assets)


class GMM(Classification):
    """
    Gaussian Mixture Model for classifying market regimes. This class extends
    the Classification class and applies a Bayesian Gaussian Mixture Model to
    classify historical asset returns into different market regimes.

    Attributes:
        n_components (int): Number of mixture components for the GMM.
        n_init (int): Number of initializations for the GMM algorithm. 
        inlier_pct (float): Percentage of datapoints labeled as inliers. Outlier observations are are not fitted upon.

    Methods:
        fit: Fits the GMM to the return data and classifies it into different regimes.
        assign: Extracts regime characteristics such as means, covariances, and weights.
    """

    def __init__(self, returns, n_components=3, n_init=100, inlier_pct=0.99):
        self.n_components = n_components
        self.n_init = n_init
        super().__init__(returns, inlier_pct)

    def fit(self):
        self.determineOutliers()

        gmm = BayesianGaussianMixture(n_components=self.n_components, n_init=self.n_init, init_params='random_from_data')
        self.gmm = gmm.fit(self.returns[self.inliers])
        self.regimes = pd.Series(gmm.predict(self.returns), index=self.returns.index, name='Regimes')

    def assign(self):
        self.means = pd.DataFrame({regime: self.gmm.means_[regime] for regime in range(self.gmm.n_components)}, index=self.returns.columns)
        self.covs = {
            regime: pd.DataFrame(self.gmm.covariances_[regime], index=self.returns.columns, columns=self.returns.columns)
            for regime in range(self.gmm.n_components)
        }
        self.regime_weights = pd.Series(self.gmm.weights_, index=range(self.gmm.n_components), name='Weights')


class HDBScan(Classification):
    """
    Hierarchical Clustering Classification for market regimes. This class extends
    the classification class and applies HDBSCAN to historical asset returns to classify
    and predict asset reutrns during each period.

    Attributes:
        min_cluster_size (int): Minimum number of observations for to classify a regime.

    Methods:
        fit: Fits the GMM to the return data and classifies it into different regimes.
    """

    def __init__(self, returns, min_cluster_size):
        self.min_cluster_size
        super().__init__(returns)

    def fit(self):
        dbscan = HDBSCAN(returns, min_cluster_size)
        self.dbscan = gmm.fit(self.returns)
        self.regimes = pd.Series(dbscan.predict(self.returns), index=self.returns.index, name='Regimes')
    
    