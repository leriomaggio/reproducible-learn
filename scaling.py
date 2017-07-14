"""Module containing all supported
feature scaling methods.

This module defines all the objects and 
functions to be used in the settings 
related to "feature scaling".
"""
from sklearn.preprocessing import (Normalizer, StandardScaler, MinMaxScaler)


# Aliases - to support string values in settings!
std = STD = StandardScaler
minmax = MINMAX = MinMaxScaler
norm = NORM = Normalizer
