# ============================================
# A. DAP Section
# ============================================

# ---------------------------
# 0. Cross Validation Section
# ---------------------------

# No. Repetitions
Cv_N = 10
# No. Fold
Cv_K = 5

# Apply Stratification to labels when
# generating folds.
stratified = True

# Enable/Disable Use of Random Labels
# If True, a random shuffling to training set labels
# will be applied.
use_random_labels = False

# Categorical Labels:
# Decide whether to apply one-hot-encode to labels
to_categorical = True

# --------------------------
# 1. Feature Scaling Section
# --------------------------

# Enable/Disable Feature Scaling
apply_feature_scaling = True

# Feature Scaling method
# ----------------------
# This can either be a string or
# an actual sklearn Transformer object
# (see sklearn.preprocessing)
from .scaling import StandardScaler
feature_scaler = StandardScaler(copy=False)

# --------------------------
# 2. Feature Ranking Section
# --------------------------

# This can be eitehr a string or a
# (custom) function object,
from .ranking import kbest_ranking
feature_ranker = kbest_ranking

# -----------------
# 2.1 Feature Steps
# _________________

# Ranges (expressed as percentage wrt. the total)
# of features to consider when generating feature steps

# Default: 5%, 25%, 50%, 75%, 100% (all)
# feature_ranges = [5, 25, 50, 75, 100]
feature_ranges = [25, 50, 75, 100]

# Include top feature in the feature steps
use_top_feature = False

# ---------
# 2.2 BORDA
# ---------

# Determine the features to use in testing through BORDA
# ranking instead of using the usual ranking feature procedure
use_borda = True


# --------------------------
# 3. Validation section
# --------------------------

# Validation in test
# ------------------
# Percentage of test data to be used as validation
# in training during the fit of the best model
validation_split_in_test = 0.2

# =================================================
# B. Machine Learning Models HyperParameter Section
# =================================================

# Use of scikit Pipelines!!
