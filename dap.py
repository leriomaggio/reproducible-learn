"""Main module implementing Data Analysis Plan.

This module provides two main classes, namely `DAP` and `DAPRegr` specifically designed for
classification and regression applications of the Daa Analysis Protocol.
"""

import os
import pickle
from abc import ABC, abstractmethod
from inspect import isclass, isfunction, ismethod

from mlpy import bootstrap_ci as mlpy_bootstrap_ci
from mlpy import cv_kfold as mlpy_cv_kfold
from mlpy import borda_count as mlpy_borda_count
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from . import settings
from .metrics import (npv, ppv, sensitivity, specificity,
                      KCCC_discrete, dor, accuracy)

# Import for Parallel Execution of DAP
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection._validation import _fit_and_score


class DAP(ABC):
    """
    Abstract Data Analysis Plan
    
    Provides a unified framework wherein execute the DAP.
    This is an abstract class so all all the abstract methods needs to be reimplemented
    in subclasses in order to work.

    """

    # Metrics Keys
    ACC = 'ACC'
    DOR = 'DOR'
    AUC = 'AUC'
    MCC = 'MCC'
    SPEC = 'SPEC'
    SENS = 'SENS'
    PPV = 'PPV'
    NPV = 'NPV'
    RANKINGS = 'ranking'
    PREDS = 'PREDS'

    BASE_METRICS = [ACC, DOR, AUC, MCC, SPEC, SENS, PPV, NPV, RANKINGS, PREDS]

    REF_STEP_METRIC = MCC

    # Confidence Intervals
    MCC_CI = 'MCC_CI'
    SPEC_CI = 'SPEC_CI'
    SENS_CI = 'SENS_CI'
    PPV_CI = 'PPV_CI'
    NPV_CI = 'NPV_CI'
    ACC_CI = 'ACC_CI'
    DOR_CI = 'DOR_CI'
    AUC_CI = 'AUC_CI'

    CI_METRICS = [MCC_CI, SPEC_CI, SENS_CI, DOR_CI, ACC_CI, AUC_CI, PPV_CI, NPV_CI]

    # Reference Metric to consider when getting best feature results (see _train_best_model)
    DAP_REFERENCE_METRIC = MCC_CI

    TEST_SET = 'TEST_SET'

    def __init__(self, experiment):
        """
        Parameters
        ----------
        experiment: sklearn.dataset.base.Bunch
            A bunch (dictionary-like) objects embedding 
            all data and corresponding attributes related 
            to the current experiment.
            
        """
        self.experiment_data = experiment

        # Map DAP configurations from settings to class attributes
        self.cv_k = settings.Cv_K
        self.cv_n = settings.Cv_N

        self.feature_ranker = settings.feature_ranker
        self.feature_scaler = settings.feature_scaler

        self.random_labels = settings.use_random_labels
        self.is_stratified = settings.stratified
        self.to_categorical = settings.to_categorical

        self.apply_feature_scaling = settings.apply_feature_scaling

        self.iteration_steps = self.cv_k * self.cv_n
        self.feature_steps = len(settings.feature_ranges)
        if settings.use_top_feature:
            self.feature_steps += 1

        # Initialise Metrics Arrays
        self.metrics = self._prepare_metrics_array()

        # Set Training Dataset
        self._set_training_data()

        # Set Test Dataset
        self._set_test_data()

        if self.random_labels:
            np.random.shuffle(self.y)

        # Model is forced to None.
        # The `create_ml_model` abstract method must be implemented to return a new
        # ML model to be used in the `fit_predict` step.
        self.ml_model_ = None

        # Store reference to the best feature ranking
        self._best_feature_ranking = None

        # -----------------------
        # Contextual Information:
        # -----------------------
        # Attributes saving information on the context of the DAP process,
        # e.g. the reference to the iteration no. of the CV,
        # the current feature step, and so on.
        #
        # This information will be updated throughout the execution of the process,
        # keeping track of current actual progresses.
        self._fold_training_indices = None  # indices of samples in training set
        self._fold_validation_indices = None  # indices of samples in validation set
        self._iteration_step_nb = -1
        self._feature_step_nb = -1
        self._runstep_nb = -1
        self._fold_nb = -1

        # Store the number of features in each iteration/feature step.
        # Note: This attribute is not really used in this general DAP process,
        # although this is paramount for its DeepLearning extension.
        # In fact it is mandatory to know
        # the total number of features to be used so to properly
        # set the shape of the first InputLayer(s).
        self._nb_features = -1

    # ====== Abstract Interface ======
    #

    @abstractmethod
    def create_ml_model(self):
        """Instantiate a new Machine Learning model to be used in the fit-predict step.
        Most likely this function has to simply call the constructor of the
        `sklearn.Estimator` object to be used.

        Examples:
        ---------

        from sklearn.svm import SVC
        return SVC(kernel='rbf', C=0.01)
        """

    @property
    @abstractmethod
    def results_folder(self):
        """Return the path to the folder where results will be stored.
        This method is abstract as its implementation is very experiment-dependent!

        Returns
        -------
        str : Path to the output folder
        """

    @property
    @abstractmethod
    def ml_model_name(self):
        """Abstract property for machine learning model associated to DAP instance.
        Each subclass should implement this property, if needed."""

    # ====== Public Interface ======
    #

    @property
    def ml_model(self):
        """Machine Learning Model to be used in DAP."""
        if not self.ml_model_:
            self.ml_model_ = self.create_ml_model()
        return self.ml_model_

    @property
    def feature_scaling_name(self):
        return self._get_label(self.feature_scaler)

    @property
    def feature_ranking_name(self):
        return self._get_label(self.feature_ranker)

    def experiment_setup(self):
        """Hook method to be implemented in case extra operations are needed before the
        experiment is started.

        Note: in future versions of the DAP, the default implementation of this method
        will establish a connection to the database.
        """
        pass

    def experiment_teardown(self):
        """Hook method to be implemented in case extra operations must be performed
         at the end of the experiment.

         Note: in future versions of the DAP, the default implementation of this method
         will close a connection to the database.
         """
        pass

    def iteration_setup(self):
        """Hook method to be implemented in case extra operations are needed before
        each iteration of the cross-validation is run."""
        pass

    def iteration_teardown(self):
        """
        Hook method to be implemented in case extra operations are needed at the end of
        each iteration of the corss-validation."""
        pass

    def fold_setup(self):
        """Hook method to be implemented in case extra operations are needed before
        each fold is entered."""
        pass

    def fold_teardown(self):
        """Hook method to be implemented in case extra operations are needed after each
        fold has been processed."""
        pass

    def feature_step_setup(self):
        """Hook method to be implemented in case extra operations are needed before
        each feature step is being processed."""
        pass

    def feature_step_teardown(self):
        """Hook method to be implemented in case extra operations are needed after
        each feature step has been processed."""
        pass

    def run(self, verbose=False):
        """
        Implement the entire Data Analysis Plan Main loop.

        Parameters
        ----------
        verbose: bool
            Flag specifying verbosity level (default: False, i.e. no output produced)

        Returns
        -------
        dap_model
            The estimator object fit on the whole training set (i.e. (self.X, self.y) )
            Note: The type or returned `dap_model` may change depending on
            different DAP subclasses (e.g. A `keras.models.Model` is returned by the
            `DeepLearningDap` subclass).

        """
        base_output_folder = self.results_folder

        # Set the different feature-steps to be used during the CV
        k_features_indices = self._generate_feature_steps(self.experiment_data.nb_features)

        # Setup Experiment
        self.experiment_setup()

        for runstep in range(self.cv_n):

            # Save contextual information
            self._runstep_nb = runstep

            # Setup Iteration
            self.iteration_setup()

            # 1. Generate K-Folds
            if self.is_stratified:
                kfold_indices = mlpy_cv_kfold(n=self.experiment_data.nb_samples,
                                              k=self.cv_k, strat=self.y, seed=runstep)
            else:
                kfold_indices = mlpy_cv_kfold(n=self.experiment_data.nb_samples,
                                              k=self.cv_k, seed=runstep)

            if verbose:
                print('=' * 80)
                print('{} over {} experiments'.format(runstep + 1, self.cv_n))
                print('=' * 80)

            for fold, (training_indices, validation_indices) in enumerate(kfold_indices):
                # Save contextual information
                self._iteration_step_nb = runstep * self.cv_k + fold
                self._fold_nb = fold
                self._fold_training_indices = training_indices
                self._fold_validation_indices = validation_indices

                # Setup fold
                self.fold_setup()

                if verbose:
                    print('=' * 80)
                    print('Experiment: {} - Fold {} over {} folds'.format(runstep + 1, fold + 1, self.cv_k))

                # 2. Split data in Training and Validation sets
                (X_train, X_validation), (y_train, y_validation) = self._train_validation_split(training_indices,
                                                                                                validation_indices)

                # 2.1 Apply Feature Scaling (if needed)
                if self.apply_feature_scaling:
                    if verbose:
                        print('-- centering and normalization using: {}'.format(self.feature_scaling_name))
                    X_train, X_validation = self._apply_scaling(X_train, X_validation)

                # 3. Apply Feature Ranking
                if verbose:
                    print('-- ranking the features using: {}'.format(self.feature_ranking_name))
                    print('=' * 80)
                ranking = self._apply_feature_ranking(X_train, y_train, seed=runstep)
                self.metrics[self.RANKINGS][self._iteration_step_nb] = ranking  # store ranking

                # 4. Iterate over Feature Steps
                for step, nb_features in enumerate(k_features_indices):

                    # Setup feature Step
                    self.feature_step_setup()

                    # 4.1 Select Ranked Features
                    X_train_fs, X_val_fs = self._select_ranked_features(ranking[:nb_features], X_train, X_validation)

                    # Store contextual info about current number of features used in this iteration and
                    # corresponding feature step.

                    # Note: The former will be effectively used in the `DeepLearningDAP` subclass to
                    # properly instantiate the Keras `InputLayer`.
                    self._nb_features = nb_features
                    self._feature_step_nb = step

                    # 5. Fit and Predict\
                    model = self.ml_model
                    # 5.1 Train the model and generate predictions (inference)
                    predictions, extra_metrics = self._fit_predict(model, X_train_fs, y_train, X_val_fs, y_validation)
                    self._compute_step_metrics(validation_indices, y_validation, predictions, **extra_metrics)

                    if verbose:
                        print(self._print_ref_step_metric())

                    # Teardown Feature Step
                    self.feature_step_teardown()

                # Teardown Fold
                self.fold_teardown()

            # Teardown Iteration
            self.iteration_teardown()

        # Compute Confidence Intervals for all target metrics
        self._compute_metrics_confidence_intervals(k_features_indices)
        # Save All Metrics to File
        self._save_all_metrics_to_file(base_output_folder, k_features_indices,
                                       self.experiment_data.feature_names, self.experiment_data.sample_names)

        if verbose:
            print('=' * 80)
            print('Fitting and predict best model')
            print('=' * 80)

        dap_model, extra_metrics = self._train_best_model(k_features_indices, seed=self.cv_n + 1)
        if extra_metrics:
            self._compute_extra_step_metrics(extra_metrics)

        # Tear down Experiment
        self.experiment_teardown()

        # Finally return the trained model
        return dap_model

    def predict_on_test(self, best_model):
        """
        Execute the last step of the DAP. A prediction using the best model
        trained in the main loop and the best number of features.

        Parameters
        ----------
        best_model
            The best model trained by the run() method
        """

        X_test = self.X_test
        Y_test = self.y_test

        if self.apply_feature_scaling:
            _, X_test = self._apply_scaling(self.X, self.X_test)

        # Select the correct features and prepare the data before predict
        feature_ranking = self._best_feature_ranking[:self._nb_features]
        X_test = self._select_ranked_features(feature_ranking, X_test)
        X_test = self._prepare_data(X_test)
        Y = self._prepare_targets(Y_test)

        predictions = self._predict(best_model, X_test)
        self._compute_test_metrics(Y_test, predictions)
        self._save_test_metrics_to_file(self.results_folder)

    def save_configuration(self):
        """
        Method that saves the configuration of the dap as a pickle. If more configuration
        need to be saved this can be done reimplementing the _save_extra_configuration method
        """

        settings_directives = dir(settings)
        settings_conf = {key: getattr(settings, key) for key in settings_directives if not key.startswith('__')}
        dump_filepath = os.path.join(self.results_folder, 'dap_settings.pickle')
        with open(dump_filepath, "wb") as dump_file:
            pickle.dumps(settings_conf)
            pickle.dump(obj=settings_conf, file=dump_file, protocol=pickle.HIGHEST_PROTOCOL)
        self._save_extra_configuration()

    # ====== Utility Methods (a.k.a. Private Interface) =======
    #

    def _get_label(self, attribute):
        """Generate a (lowercase) label referring to the provided
         attribute object. This method is used in all class properties
         requiring to return the name of specific methods used in DAP steps,
         most likely to be included in reports and logs.
        """
        if isinstance(attribute, str):
            name = attribute
        elif isclass(attribute) or isfunction(attribute):
            name = attribute.__name__
        elif ismethod(attribute):
            name = attribute.__qualname__
        else:
            name = attribute.__class__.__name__
        return name.lower()

    def _set_training_data(self):
        """Default implementation for classic and quite standard DAP implementation.
         More complex implementation require overriding this method.
        """
        self.X = self.experiment_data.training_data
        self.y = self.experiment_data.targets

    def _set_test_data(self):
        self.X_test = self.experiment_data.test_data
        self.y_test = self.experiment_data.test_targets

    def _prepare_metrics_array(self):
        """
        Initialise Base "Standard" DAP Metrics to be monitored and saved during the 
        data analysis plan.
        """

        metrics_shape = (self.iteration_steps, self.feature_steps)
        metrics = {
            self.RANKINGS: np.empty((self.iteration_steps, self.experiment_data.nb_features), dtype=np.int),

            self.NPV: np.empty(metrics_shape),
            self.PPV: np.empty(metrics_shape),
            self.SENS: np.empty(metrics_shape),
            self.SPEC: np.empty(metrics_shape),
            self.MCC: np.empty(metrics_shape),
            self.AUC: np.empty(metrics_shape),
            self.ACC: np.empty(metrics_shape),
            self.DOR: np.empty(metrics_shape),

            self.PREDS: np.empty(metrics_shape + (self.experiment_data.nb_samples,), dtype=np.int),

            # Confidence Interval Metrics-specific
            # all entry are assumed to have the following structure
            # (mean, lower_bound, upper_bound)
            self.MCC_CI: np.empty((self.feature_steps, 3)),
            self.ACC_CI: np.empty((self.feature_steps, 3)),
            self.AUC_CI: np.empty((self.feature_steps, 3)),
            self.PPV_CI: np.empty((self.feature_steps, 3)),
            self.NPV_CI: np.empty((self.feature_steps, 3)),
            self.SENS_CI: np.empty((self.feature_steps, 3)),
            self.SPEC_CI: np.empty((self.feature_steps, 3)),
            self.DOR_CI: np.empty((self.feature_steps, 3)),

            # Test dictionary
            self.TEST_SET: dict()
        }
        # Initialize to Flag Values
        metrics[self.PREDS][:, :, :] = -10
        return metrics

    def _compute_step_metrics(self, validation_indices, y_true_validation,
                              predictions, **extra_metrics):
        """
        Compute the "classic" DAP Step metrics for corresponding iteration-step and feature-step.
        
        Parameters
        ----------
        validation_indices: array-like, shape = [n_samples]
            Indices of validation samples
            
        y_true_validation: array-like, shape = [n_samples]
            Array of true targets for samples in the validation set
            
        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_fit_predict` method.
            
        Other Parameters
        ----------------
            
        extra_metrics:
            List of extra metrics to log during execution returned by the `_fit_predict` method
            This list will be processed separately from standard "base" metrics.
        """

        # Process predictions
        predicted_labels, _ = predictions  # prediction_probabilities are not used for base metrics.

        # Compute Base Step Metrics
        iteration_step, feature_step = self._iteration_step_nb, self._feature_step_nb

        self.metrics[self.PREDS][iteration_step, feature_step, validation_indices] = predicted_labels
        self.metrics[self.NPV][iteration_step, feature_step] = npv(y_true_validation, predicted_labels)
        self.metrics[self.PPV][iteration_step, feature_step] = ppv(y_true_validation, predicted_labels)
        self.metrics[self.SENS][iteration_step, feature_step] = sensitivity(y_true_validation, predicted_labels)
        self.metrics[self.SPEC][iteration_step, feature_step] = specificity(y_true_validation, predicted_labels)
        self.metrics[self.MCC][iteration_step, feature_step] = KCCC_discrete(y_true_validation, predicted_labels)
        self.metrics[self.AUC][iteration_step, feature_step] = roc_auc_score(y_true_validation, predicted_labels)
        self.metrics[self.DOR][iteration_step, feature_step] = dor(y_true_validation, predicted_labels)
        self.metrics[self.ACC][iteration_step, feature_step] = accuracy(y_true_validation, predicted_labels)

        if extra_metrics:
            self._compute_extra_step_metrics(validation_indices, y_true_validation,
                                             predictions, **extra_metrics)

    def _compute_extra_step_metrics(self, validation_indices=None, y_true_validation=None,
                                    predictions=None, **extra_metrics):
        """Method to be implemented in case extra metrics are returned during the *fit-predict* step.
           By default, no additional extra metrics are returned.

        Parameters are all the same of the "default" `_compute_step_metrics` method, with the
        exception that parameters can be `None` to make the API even more flexible!

        """
        pass

    def _compute_test_metrics(self, y_true_test, predictions, **extra_metrics):

        """
        Compute the "classic" DAP Step metrics for test data

        Parameters
        ----------

        y_true_test: array-like, shape = [n_samples]
            Array of true targets for samples in the test set

        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_predict` method.

        Other Parameters
        ----------------

        extra_metrics:
            List of extra metrics to log during execution returned by the `predict` method
            This list will be processed separately from standard "base" metrics.
        """

        # Process predictions
        predicted_labels, _ = predictions  # prediction_probabilities are not used for base metrics.

        self.metrics[self.TEST_SET][self.PREDS] = predicted_labels
        self.metrics[self.TEST_SET][self.NPV] = npv(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.PPV] = ppv(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.SENS] = sensitivity(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.SPEC] = specificity(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.MCC] = KCCC_discrete(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.AUC] = roc_auc_score(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.DOR] = dor(y_true_test, predicted_labels)
        self.metrics[self.TEST_SET][self.ACC] = accuracy(y_true_test, predicted_labels)

        if extra_metrics:
            self._compute_extra_test_metrics(y_true_test, predictions, **extra_metrics)

    def _compute_extra_test_metrics(self, y_true_test=None, predictions=None, **extra_metrics):
        """Method to be implemented in case extra metrics are returned during the *predict* step.
           By default, no additional extra metrics are returned.

        Parameters are all the same of the "default" `_compute_test_metrics` method, with the
        exception that can be `None` to make the API even more flexible!
        """
        pass

    # Compute Confidence Intervals for Metrics
    def _compute_ci_metric(self, feature_steps, ci_metric_key, metric_key):
        """

        Parameters
        ----------
        feature_steps: list
            List of feature steps considered during the DAP process
        ci_metric_key: str
            The name of the key of metrics array where to store Metric-CI values
        metric_key: str
            The name of the reference metric used to calculate
            Confidence Intervals
        """
        metric_means = np.mean(self.metrics[metric_key], axis=0)
        for step, _ in enumerate(feature_steps):
            metric_mean = metric_means[step]
            ci_low, ci_hi = mlpy_bootstrap_ci(self.metrics[metric_key][:, step])
            self.metrics[ci_metric_key][step] = np.array([metric_mean, ci_low, ci_hi])

    def _compute_metrics_confidence_intervals(self, feature_steps):
        """Compute Confidence Intervals for all target metrics.
        
        Parameters
        ----------
        feature_steps: list
            List of feature steps considered during the DAP process
        """

        # Target metrics are all those included in the CI_METRICS list.
        for ci_key in self.CI_METRICS:
            metric_key = getattr(self, ci_key.split('_')[0])
            self._compute_ci_metric(feature_steps, ci_key, metric_key)

    def _print_ref_step_metric(self):
        return "{}: {}".format(self.REF_STEP_METRIC,
                               self.metrics[self.REF_STEP_METRIC][self._iteration_step_nb, self._feature_step_nb])

    @staticmethod
    def _save_metric_to_file(metric_filename, metric, columns=None, index=None):
        """
        Write single metric data to a CSV file (tab separated).
        Before writing data to files, data are converted to a `pandas.DataFrame`.
        
        Parameters
        ----------
        metric_filename: str
            The name of the output file where to save metrics data
            
        metric: array-like, shape = (n_iterations, n_feature_steps) [typically]
            The 2D data for input metric collected during the whole DAP process.
            
        columns: list
            List of labels for columns of the data frame (1st line in the output file).
            
        index: list
            List of labels to be used as Index in the DataFrame.
        """

        df = pd.DataFrame(metric, columns=columns, index=index)
        df.to_csv(metric_filename, sep='\t')

    def _save_all_metrics_to_file(self, base_output_folder_path, feature_steps, feature_names, sample_names):
        """
        Save all basic metrics to corresponding files.
        
        Parameters
        ----------
        base_output_folder_path: str
            Path to the output folder where files will be saved.
             
        feature_steps: list
            List with all feature steps.
            
        feature_names: list
            List with all the names of the features
            
        sample_names: list
            List of all sample names
        """

        blacklist = [self.RANKINGS, self.PREDS]
        for key in self.BASE_METRICS:
            if key not in blacklist:
                metric_values = self.metrics[key]
                self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_{}.txt'.format(key)),
                                          metric_values, feature_steps)

        # Save Ranking
        self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_{}.txt'.format(self.RANKINGS)),
                                  self.metrics[self.RANKINGS], feature_names)

        # Save Metrics for Predictions
        for i_step, step in enumerate(feature_steps):
            for key in blacklist[1:]:  # exclude RANKING, already saved
                self._save_metric_to_file(os.path.join(base_output_folder_path,
                                                       'metric_{}_fs{}.txt'.format(key, step)),
                                          self.metrics[key][:, i_step, :], sample_names)

        # Save Confidence Intervals Metrics
        # NOTE: All metrics will be saved together into a unique file.
        ci_metric_values = list()
        metric_names = list()  # collect names to become indices of resulting pd.DataFrame
        for metric_key in self.CI_METRICS:
            metric_values = self.metrics[metric_key]
            nb_features_steps = metric_values.shape[0]
            metric_names.extend(['{}-{}'.format(metric_key, s) for s in range(nb_features_steps)])
            ci_metric_values.append(metric_values)
        ci_metric_values = np.vstack(ci_metric_values)
        self._save_metric_to_file(os.path.join(base_output_folder_path, 'CI_All_metrics.txt'),
                                  ci_metric_values, columns=['Mean', 'Lower', 'Upper'],
                                  index=metric_names)

    def _save_test_metrics_to_file(self, base_output_folder_path):
        """
        Save all basic metrics to corresponding files.

        Parameters
        ----------
        base_output_folder_path: str
            Path to the output folder where files will be saved.

        """

        self._save_metric_to_file(os.path.join(base_output_folder_path, 'test_PRED.txt'),
                                  self.metrics[self.TEST_SET][self.PREDS], columns=['prediction'])

        column_names = []
        self.metrics[self.TEST_SET].pop(self.PREDS)
        test_metrics = np.zeros((1, len(self.metrics[self.TEST_SET].keys())))
        for i, item in enumerate(self.metrics[self.TEST_SET].items()):
            key, value = item
            test_metrics[0, i] = value
            column_names.append(key)
        self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_test.txt'),
                                  test_metrics, column_names)

    @staticmethod
    def _generate_feature_steps(nb_features):
        """
        Generate the feature_steps, i.e. the total number
        of features to select at each step in the Cross validation.
        
        Total features for each step are collected according to the 
        total number of features, and feature percentages specified in settings
        (see: settings.feature_ranges).
        
        Parameters
        ----------
        nb_features: int
            The total number of feature
            
        Returns
        -------
        k_features_indices: list
            The number of features (i.e. indices) to consider 
            in slicing features at each (feature) step.
        """
        k_features_indices = list()
        if settings.use_top_feature:
            k_features_indices.append(1)
        for percentage in settings.feature_ranges:
            k = np.ceil((nb_features * percentage) / 100).astype(np.int)
            k_features_indices.append(k)

        return k_features_indices

    def _save_extra_configuration(self):
        """
        Method to be implemented in case that extra configurations must be saved. 
        By default, no extra configurations are saved.
        """
        pass

    # ====== DAP Ops Methods ======
    #
    def _train_validation_split(self, training_indices, validation_indices):
        """DAP standard implementation for train-validation splitting

        Parameters
        ----------
        training_indices: array-like
            Indices of samples in training set

        validation_indices: array-like
            Indices of samples in the validation set
        """
        Xs_tr, Xs_val = self.X[training_indices], self.X[validation_indices]
        ys_tr, ys_val = self.y[training_indices], self.y[validation_indices]
        return (Xs_tr, Xs_val), (ys_tr, ys_val)

    def _get_feature_scaler(self):
        """
        Retrieves the Scaler object corresponding to what 
        specified in settings.
        
        Returns
        -------    
        sklearn.base.TransformerMixin
            Scikit-learn feature scaling estimator, already trained on
            input training data, and so ready to be applied on validation data.
            
        Raises
        ------
        ValueError if a wrong/not supported feature scaling method has been
        specified in settings.
        """
        # This mimics what is done by Keras when accepting
        # string parameters (e.g. `optimizer='los'`).
        try:
            self.feature_scaler = globals().get(self.feature_scaler)
        except:
            raise ValueError('No Feature Scaling Method found!')

    def _apply_scaling(self, X_train, X_validation):
        """
        Apply feature scaling on training and validation data.
        
        Parameters
        ----------
        X_train: array-like, shape = (n_samples, n_features)
            Training data to `fit_transform` by the selected feature scaling method
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data to `transform` by the selected feature scaling method

        Returns
        -------
        X_train_scaled: array-like, shape = (n_samples, n_features)
            Training data with features scaled according to the selected
            scaling method/estimator.
            
        X_val_scaled: array-like, shape = (n_samples, n_features)
            Validation data with features scaled according to the selected
            scaling method/estimator.
        
        """
        if isinstance(self.feature_scaler, str):
            scaler = self._get_feature_scaler()
        else:
            scaler = self.feature_scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_validation)
        return X_train_scaled, X_val_scaled

    def _apply_feature_ranking(self, X_train, y_train, seed=np.random.seed(1234)):
        """
        Compute the ranking of the features as required. It raises an exception 
        if a wrong rank_method name is specified in settings. 
         
        Parameters
        ----------
        X_train: array-like, shape = (n_samples, n_features)
            Training data matrix
            
        y_train: array-like, shape = (n_samples)
            Training labels
            
        seed: int (default: np.random.rand(1234))
            Integer seed to use in random number generators.

        Returns
        -------
        ranking: array-like, shape = (n_features, )
            features ranked by the selected ranking method
        """
        if isinstance(self.feature_scaler, str):
            # This mimics what is done by Keras when accepting
            # string parameters (e.g. `optimizer='los'`).
            try:
                self.feature_ranker = globals().get(self.feature_ranker)
            except:
                raise ValueError('No Feature Ranking Method found!')

        ranking = self.feature_ranker(X_train, y_train, seed=seed)
        return ranking

    def _select_ranked_features(self, ranked_feature_indices, X_train, X_validation=None):
        """Filter features according to input ranking
        
        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features]
            Training data
        
        X_validation: array-like, shape = [n_samples, n_features]
            Validation data
            
        ranked_feature_indices: X_train: array-like, shape = [n_features_step]
            Array of indices corresponding to features to select.
        
        Returns
        -------
        X_train: array-like, shape = [n_samples, n_features_step]
            Training data with selected features
        
        X_validation: array-like, shape = [n_samples, n_features_step]
            Validation data with selected features
        """

        X_train_fs = X_train[:, ranked_feature_indices]
        if X_validation is not None:
            X_val_fs = X_validation[:, ranked_feature_indices]
            return X_train_fs, X_val_fs
        return X_train_fs

    def _fit_predict(self, model, X_train, y_train, X_validation, y_validation=None):
        """
        Core method to generate metrics on (feature-step) data by fitting 
        the input machine learning model and predicting on validation data.
        on validation data.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Scikit-learn Estimator object
            
        X_train: array-like, shape = (n_samples, n_features)
            Training data of the current feature step
            
        y_train: array-like, shape = (n_samples, )
            Training targets
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data of the current feature step
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation targets (None by default as it is not used in predict)

        Returns
        -------
        predictions: array-like, shape = (n_samples, ) or tuple of array-like objects.
            Array containing the predictions generated by the model. What is actually
            contained in the `prediction` array-like object is strongly
            related to the task at hand (see `_predict` method)
            
        extra_metrics:
            List of extra metrics to log during execution.
        """

        X_train = self._prepare_data(X_train, training_data=True)
        y_train = self._prepare_targets(y_train, training_labels=True)
        model, extra_metrics = self._fit(model, X_train, y_train)

        X_validation = self._prepare_data(X_validation, training_data=False)
        predictions = self._predict(model, X_validation)
        return predictions, extra_metrics

    def _prepare_data(self, X, training_data=True):
        """
        Preparation of data before training/inference.
        Current implementation (default behaviour) does not apply
        any operation (i.e. Input data remains unchanged!)
        
        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Input data to prepare
            
        training_data: bool (default: True)
            Flag indicating whether input data are training data or not.
            This flag is included as it may be required to prepare data
            differently depending they're training or validation data.

        Returns
        -------
        array-like, shape = (n_samples, n_features)
            Input data unchanged (Identity)
        """
        return X

    def _prepare_targets(self, y, training_labels=True):
        """
        Preparation of targets before training/inference.
        Current implementation only checks whether categorical one-hot encoding 
        is required on labels. Otherwise, input labels remain unchanged. 
        
        Parameters
        ----------
        y: array-like, shape = (n_samples, )
            array of targets for each sample.
            
        training_labels: bool (default: True)
            Flag indicating whether input targets refers to training data or not.
            This flag is included as it may be required to prepare labels
            differently depending they refers to training or validation data.

        Returns
        -------
        y : array-like, shape = (n_samples, )
            Array of targets whose dimensions will be unchanged, if no encoding has been applied),
            or (samples x nb_classes) otherwise.

        """
        if self.to_categorical:
            if self.experiment_data.nb_classes == 2:  # binary problem
                # Trick to calculate one-hot-encoding using sklearn function
                # This is because in the new version of Scikit-learn, the label_binarize
                # function returns an output shape of [n_samples, 1] for Binary problems 
                y = label_binarize(y, classes=np.arange(self.experiment_data.nb_classes + 1))
                # output shape of y will be [n_samples, 3], so we drop one dimension of all zeros
                y = y[:, :2]
            else:
                y = label_binarize(y, classes=np.arange(self.experiment_data.nb_classes))
            y = y.astype(np.float32)  # Keras default - consider moving this to Deep Learning DAP
        return y

    def _fit(self, model, X_train, y_train, X_validation=None, y_validation=None):
        """
        Default implementation of the training (`fit`) step of input model 
        considering scikit-learn Estimator API.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Generic Scikit-learn Classifier
            
        X_train: array-like, shape = (n_samples, n_features)
            Training data
            
        y_train: array-like, shape = (n_samples, )
            Training labels
            
        X_validation: array-like, shape = (n_samples, n_features) - default: None
            Validation data to be used in combination with Training data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels to be used in combination with validation data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.

        Returns
        -------
        model:    the trained model
        
        extra_metrics: dict
            List of extra metrics to be logged during execution. Default: None
        """
        model = model.fit(X_train, y_train)
        extra_metrics = dict()  # No extra metrics is returned by default
        return model, extra_metrics

    def _predict(self, model, X_validation, y_validation=None, **kwargs):
        """
        Default implementation of the inference (`predict`) step of input model 
        considering scikit-learn Estimator API.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Classifier sklearn model implementing Estimator API
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels. None by default as it is not used.
            
        Other Parameters
        ----------------   
        
        kwargs: dict
            Additional arguments to pass to the inference

        Returns
        -------
        predicted_classes: array-like, shape = (n_samples, )
            Array containing the class predictions generated by the model
            
        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            Array containing the prediction probabilities estimated by the model 
            for each of the considered targets (i.e. classes)
        """
        predicted_classes = model.predict(X_validation)
        if hasattr(model, 'predict_proba'):
            predicted_class_probs = model.predict_proba(X_validation)
        else:
            predicted_class_probs = None
        return (predicted_classes, predicted_class_probs)

    def _train_best_model(self, k_feature_indices, seed=None):
        """
        Train a new model on the best set of features resulting **after** the entire
        Cross validation process has been completed.
        
        Parameters
        ----------
        k_feature_indices: list
            List containing the total number of features to consider at each 
            feature-step.
            
        seed: int
            Random seed to be used for new feature ranking.

        Returns
        -------
        dap_model: sklearn.base.BaseEstimator
            The estimator object fit on the whole training set (i.e. (self.X, self.y) )
            
        extra_metrics: dict
            Dictionary containing information about extra metrics to be monitored during the 
            process and so to be saved, afterwards.
        """

        # Get Best Feature Step (i.e. no. of features to use)
        reference_metric_avg_values = self.metrics[self.DAP_REFERENCE_METRIC][:, 0]
        max_index = np.argmax(reference_metric_avg_values)
        best_nb_features = k_feature_indices[max_index]

        # update contextual information
        self._nb_features = best_nb_features  # set this attr. to possibly reference the model
        self._iteration_step_nb = (self.cv_n * self.cv_k)  # last step
        self._feature_step_nb = (self.cv_n * self.cv_k)  # flag value indicating last step

        # Set Training data
        if self.is_stratified:
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y,
                                                              test_size=settings.validation_split_in_test,
                                                              random_state=np.random.seed(),
                                                              stratify=self.y)
        else:
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y,
                                                              test_size=settings.validation_split_in_test,
                                                              random_state=np.random.seed())

        # 2.1 Apply Feature Scaling (if needed)
        if self.apply_feature_scaling:
            X_train, X_val = self._apply_scaling(X_train, X_val)

        # 3. Apply Feature Ranking
        if settings.use_borda:
            ranking, _, _ = mlpy_borda_count(self.metrics[self.RANKINGS])
        else:
            ranking = self._apply_feature_ranking(X_train, y_train, seed=seed)

        self._best_feature_ranking = ranking
        Xs_train_fs, Xs_val_fs = self._select_ranked_features(ranking[:best_nb_features], X_train, X_val)

        # 4. Fit the model
        model = self.ml_model

        # 4.1 Prepare data
        Xs_train_fs = self._prepare_data(Xs_train_fs, training_data=True)
        Xs_val_fs = self._prepare_data(Xs_val_fs, training_data=False)
        y_train = self._prepare_targets(y_train, training_labels=True)
        y_val = self._prepare_targets(y_val, training_labels=False)

        model, extra_metrics = self._fit(model, Xs_train_fs, y_train, Xs_val_fs, y_val)
        return model, extra_metrics

    # ===========================================================




class DAPRegr(DAP):
    """Specialisation of the DAP implementation for Regression Tasks"""

    EVS = 'EVS'
    MAE = 'MAE'
    MSE = 'MSE'
    MedAE = 'MedAE'
    R2 = 'R2'

    EVS_CI = 'EVS_CI'
    MAE_CI = 'MAE_CI'
    MSE_CI = 'MSE_CI'
    MedAE_CI = 'MedAE_CI'
    R2_CI = 'R2_CI'

    DAP_REFERENCE_METRIC = R2_CI
    REF_STEP_METRIC = R2

    BASE_METRICS = [EVS, MAE, MSE, MedAE, R2]
    CI_METRICS = [EVS_CI, MAE_CI, MedAE_CI, MSE_CI, R2_CI]

    # ====== Utility Methods (Specialisation) =======

    def _prepare_metrics_array(self):
        """
        Initialise Base "Standard" DAP Metrics to be monitored and saved during the
        data analysis plan.
        """

        metrics_shape = (self.iteration_steps, self.feature_steps)
        metrics = {
            self.RANKINGS: np.empty((self.iteration_steps, self.experiment_data.nb_features), dtype=np.int),

            self.EVS: np.empty(metrics_shape),
            self.MAE: np.empty(metrics_shape),
            self.MSE: np.empty(metrics_shape),
            self.MedAE: np.empty(metrics_shape),
            self.R2: np.empty(metrics_shape),
            self.PREDS: np.empty(metrics_shape + (self.experiment_data.nb_samples,), dtype=np.int),

            # Confidence Interval Metrics-specific
            # all entry are assumed to have the following structure
            # (mean, lower_bound, upper_bound)
            self.EVS_CI: np.empty((self.feature_steps, 3)),
            self.MAE_CI: np.empty((self.feature_steps, 3)),
            self.MSE_CI: np.empty((self.feature_steps, 3)),
            self.MedAE_CI: np.empty((self.feature_steps, 3)),
            self.R2_CI: np.empty((self.feature_steps, 3)),

            # Test dictionary
            self.TEST_SET: dict()
        }
        # Initialize to Flag Values
        metrics[self.PREDS][:, :, :] = -10
        return metrics

    def _compute_step_metrics(self, validation_indices, y_true_validation,
                              predictions, **extra_metrics):
        """
        Compute the "classic" DAP Step metrics for corresponding iteration-step and feature-step.

        Parameters
        ----------
        validation_indices: array-like, shape = [n_samples]
            Indices of validation samples

        y_true_validation: array-like, shape = [n_samples]
            Array of true targets for samples in the validation set

        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_fit_predict` method.

        Other Parameters
        ----------------

        extra_metrics:
            List of extra metrics to log during execution returned by the `_fit_predict` method
            This list will be processed separately from standard "base" metrics.
        """

        # Process predictions
        predicted_values, _ = predictions

        # Compute Base Step Metrics
        iteration_step, feature_step = self._iteration_step_nb, self._feature_step_nb
        self.metrics[self.PREDS][iteration_step, feature_step, validation_indices] = predicted_values
        self.metrics[self.EVS][iteration_step, feature_step] = explained_variance_score(y_true_validation,
                                                                                        predicted_values)
        self.metrics[self.MAE][iteration_step, feature_step] = mean_absolute_error(y_true_validation,
                                                                                   predicted_values)
        self.metrics[self.MedAE][iteration_step, feature_step] = median_absolute_error(y_true_validation,
                                                                                       predicted_values)
        self.metrics[self.MSE][iteration_step, feature_step] = mean_squared_error(y_true_validation,
                                                                                  predicted_values)
        self.metrics[self.R2][iteration_step, feature_step] = r2_score(y_true_validation, predicted_values)

        if extra_metrics:
            self._compute_extra_step_metrics(validation_indices, y_true_validation, predictions, **extra_metrics)

    def _compute_test_metrics(self, y_true_test, predictions, **extra_metrics):

        """
        Compute the "classic" DAP Step metrics for test data

        Parameters
        ----------

        y_true_test: array-like, shape = [n_samples]
            Array of true targets for samples in the test set

        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_predict` method.

        Other Parameters
        ----------------

        extra_metrics:
            List of extra metrics to log during execution returned by the `predict` method
            This list will be processed separately from standard "base" metrics.
        """

        # Process predictions
        predicted_values, _ = predictions  # prediction_probabilities are not used for base metrics.

        self.metrics[self.TEST_SET][self.PREDS] = predicted_values
        self.metrics[self.TEST_SET][self.EVS] = explained_variance_score(y_true_test, predicted_values)
        self.metrics[self.TEST_SET][self.MAE] = mean_absolute_error(y_true_test, predicted_values)
        self.metrics[self.TEST_SET][self.MSE] = mean_squared_error(y_true_test, predicted_values)
        self.metrics[self.TEST_SET][self.MedAE] = median_absolute_error(y_true_test, predicted_values)
        self.metrics[self.TEST_SET][self.R2] = r2_score(y_true_test, predicted_values)

        if extra_metrics:
            self._compute_extra_test_metrics(y_true_test, predictions, **extra_metrics)
