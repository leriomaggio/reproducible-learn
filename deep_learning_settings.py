# =================================================
# C. Deep Learning Models Hyperparameter Section
# =================================================

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# This following section contains settings
# that will be used only by the
# `DeepLearningDAP` class!
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-==-=-=

# -------------------
# 1 Model Fit Section
# ___________________

# No. of Epochs
epochs = 200

# Size of the Batch
batch_size = 32

# Verbosity level of `fit` method
fit_verbose = 1

# Shuffle (boolean)
# -----------------
# Whether to shuffle the samples at each epoch.
shuffle = True  # Shuffle samples at each epoch, by default.

# Class Weights
# -------------
# Dictionary mapping classes to a weight value
# used for scaling the loss function (during training only).
class_weight = None

# Sample Weights
# --------------
# Numpy array of weights for the training samples,
# used for scaling the loss function (during training only).
# You can either pass a flat (1D) Numpy array with the same
# length as the input samples
# (1:1 mapping between weights and samples),
# or in the case of temporal data, you can pass a 2D array
# with shape (samples, sequence_length), to
# apply a different weight to every timestep of every sample.
# In this case you should make sure to specify
# sample_weight_mode="temporal" in compile().
sample_weight = None

# Initial Epoch
# -------------
# Epoch at which to start training
# (useful for resuming a previous training run).
initial_epoch = 0  # Default: 0 -  first epoch!

# Additional Callbacks
# --------------------
# By default, the `keras.callbacks.ModelSelection` callback
# will be applied at each fit, in addition to the default
# `keras.callbacks.History`.
#
# To automatically plug additional callbacks into
# model fit, please add configured keras Callbacks objects
# in the list below
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=4,
                           min_delta=1e-06, mode='min')]

# Validation split
# ----------------
# (Automatically ignored if `validation_data` is provided)
validation_split = 0.0  # Default: no split

# -----------------------
# 2 Model Compile Section
# _______________________

# Loss Function
# --------------
# This may be either a string or a function object
# (see keras.losses for examples)
loss='categorical_crossentropy'

# Loss Weights
# -------------
# List of weights to associate to losses
# (in case of multi-output networks)
loss_weights = None

# Additional Compile Settings
# ---------------------------
# Additional Compile settings directly
# passed to Theano functions. Ignored by Tensorflow
extra_compilation_parameters = {}

# Loss Metric(s)
# --------------
# List of metrics to optimise in the training.
# These can either be strings or function objects
# Default metric is accuracy.
# (see keras.metrics for examples)
metrics = ['accuracy']

# Optimizer
# ---------
# This can either be a string or an
# optimizer object (see keras.optimizers)
from keras.optimizers import Adam
optimizer = Adam(lr=0.001, decay=1e-06,
                 epsilon=1e-08, beta_1=0.9, beta_2 = 0.999)
