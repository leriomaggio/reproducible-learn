"""Module containing all supported
feature ranking methods.

This module defines all the objects and 
functions to be used in the settings 
related to "feature ranking".
"""
from .relief import ReliefF
from sklearn.feature_selection import SelectKBest
import numpy as np


def random_ranking(X_train, y_train, seed=None):
    """Apply Random Feature Ranking
    
    Parameters
    ----------
    X_train: array-like, shape = (n_samples, n_features)
        Training data matrix
        
    y_train: array-like, shape = (n_samples)
        Training labels
        
    seed: int (default: None)
        Integer seed to use in random number generators.
        
    Returns
    -------
    ranking: array-like, shape = (n_features, )
        Resulting ranking of features
    """
    if not seed:
        seed = np.random.seed(1234)
    ranking = np.arange(X_train.shape[1])
    np.random.seed(seed)
    return np.random.shuffle(ranking)

# Alias for Random ranking.
random = RANDOM = random_ranking


def relief_ranking(X_train, y_train, seed=None):
    """Apply RelieF based Ranking.
    
    Parameters
    ----------
    X_train: array-like, shape = (n_samples, n_features)
        Training data matrix
        
    y_train: array-like, shape = (n_samples)
        Training labels
        
    seed: int (default: None)
        Integer seed to use in random number generators.
        
    Returns
    -------
    ranking: array-like, shape = (n_features, )
        Resulting ranking of features
    """
    if not seed:
        seed = np.random.seed(1234)
    relief_model = ReliefF(k=3, seed=seed)
    relief_model.learn(X_train, y_train)
    w = relief_model.w()
    ranking = np.argsort(w)[::-1]
    return ranking

# Aliases for settings
relief = RELIEFF = Relieff = relief_ranking


def kbest_ranking(X_train, y_train, seed=None):
    """Apply SelectKBest based Ranking.
    
    Parameters
    ----------
    X_train: array-like, shape = (n_samples, n_features)
        Training data matrix
        
    y_train: array-like, shape = (n_samples)
        Training labels
        
    seed: int (default: None)
        Integer seed to use in random number generators.
        
    Returns
    -------
    ranking: array-like, shape = (n_features, )
        Resulting ranking of features
    """

    selector = SelectKBest(k=10)
    selector.fit(X_train, y_train)
    ranking = np.argsort(-np.log10(selector.pvalues_))[::-1]
    return ranking


# Aliases for settings
kbest = KBEST = kbest_ranking
