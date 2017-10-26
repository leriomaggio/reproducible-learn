import os
from abc import abstractmethod

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

from .dap import DAP, DAPRegr
from . import deep_learning_settings


class DeepLearningDAP(DAP):
    """
    DAP Specialisation for plans using Deep Neural Network Models as Learning models
    """

    # Neural Network Specific Metric Keys
    NN_VAL_ACC = 'NN_val_acc'
    NN_ACC = 'NN_acc'
    NN_VAL_LOSS = 'NN_val_loss'
    NN_LOSS = 'NN_loss'
    HISTORY = 'model_history'
    NETWORK_METRICS = [NN_VAL_ACC, NN_ACC, NN_VAL_LOSS, NN_LOSS]

    def __init__(self, experiment):
        super(DeepLearningDAP, self).__init__(experiment=experiment)

        # Set additional attributes from Deep Learning Specific Settings set.
        self.learning_epochs = deep_learning_settings.epochs
        self.batch_size = deep_learning_settings.batch_size
        self.fit_verbose = deep_learning_settings.fit_verbose
        self.fit_callbacks = deep_learning_settings.callbacks

        # extra fit parameters
        self.extra_fit_params = {
            'validation_split': deep_learning_settings.validation_split,
            'shuffle': deep_learning_settings.shuffle,
        }
        if deep_learning_settings.initial_epoch:
            self.extra_fit_params['initial_epoch'] = deep_learning_settings.initial_epoch
        if deep_learning_settings.sample_weight:
            self.extra_fit_params['sample_weight'] = deep_learning_settings.sample_weight
        if deep_learning_settings.class_weight:
            self.extra_fit_params['class_weight'] = deep_learning_settings.class_weight

        # Compilation Settings, we need to salve the class and
        # configuration since after every experiment we need to call
        # K.clear_session to reduce TensorFlow memory leak. This operation
        # destroy the optimizer and we need to recreate it. To do so we need
        # both the class and the configuration
        self.optimizer = deep_learning_settings.optimizer
        if not isinstance(self.optimizer, str):
            self.optimizer_class = self.optimizer.__class__
            self.optimizer_configuration = self.optimizer.get_config()

        self.loss_function = deep_learning_settings.loss
        self.learning_metrics = deep_learning_settings.metrics
        self.loss_weights = deep_learning_settings.loss_weights
        self.extra_compile_params = deep_learning_settings.extra_compilation_parameters

        # Force the `to_categorical` setting to True for Deep Learning DAP
        # (this is True in the 99% of the cases and avoids to bother in case one forgets
        #  to check and set the corresponding setting)
        self.to_categorical = True

        # Model Cache - one model reference per feature step
        self._model_cache = {}
        self._do_serialisation = True  # Checks whether model serialisation works

    @property
    def ml_model(self):
        """Keras Model to be used in DAP.

        Note: Differently from "standard" DAP, bound to sklearn estimators,
        a **brand new** model (network) must be returned at each call,
        namely with a brand new set of weights each time this property is called.
        """
        # if not self.ml_model_:
        #     self.ml_model_ = self.create_ml_model()
        # else:
        #     pass  # Random Shuffling of Weights

        # Note: the value of self._nb_features attribute is updated during the main DAP loop,
        # during each iteration, before this !

        cache_key = self._nb_features
        if cache_key in self._model_cache:
            if self._do_serialisation:
                try:
                    from_json = self._model_cache[cache_key]
                    model = model_from_json(from_json, custom_objects=self.custom_layers_objects())
                except Exception:
                    self._do_serialisation = False
                    self._model_cache = dict()  #reset cache
                    model = self.create_ml_model()
            else:
                model = self.create_ml_model()
        else:
            model = self.create_ml_model()
            if self._do_serialisation:
                try:
                    self._model_cache[cache_key] = model.to_json()
                except Exception:
                    # Something went wrong during serialisation
                    self._do_serialisation = False

        self.ml_model_ = model
        return self.ml_model_

    def clear_network_graph(self):
        """
        Method that resets the Keras session at the end of each experiment.
        We need this in order to reduce the memory leak from tensorflow.
        Please note that the optimizer is part of the graph so needs to
        be recreated after this call.
        """
        K.clear_session()

        # If the optimizer is not a string we need to recreate it
        # since the graph is new after calling clear session

        if not isinstance(self.optimizer, str):
            self.optimizer = self.optimizer_class(**self.optimizer_configuration)

    def create_ml_model(self):
        """Instantiate a new Keras Deep Network to be used in the fit-predict step.

        Returns
        -------
        model: keras.models.Model
            The new deep learning Keras model.
        """

        model = self._build_network()

        # Set Compilation Params
        extra_compile_params = {}
        if self.loss_weights:
            extra_compile_params['loss_weights'] = self.loss_weights
        if self.extra_compile_params:
            extra_compile_params.update(**self.extra_compile_params)

        model.compile(loss=self.loss_function, optimizer=self.optimizer,
                      metrics=self.learning_metrics, **extra_compile_params)
        return model

    @staticmethod
    def custom_layers_objects():
        """Utility method to specify references to custom layer objects,
        to be used in de-serialising models.

        Returns
        -------
        dic
            dictionary mapping names (strings) to custom classes or functions to be
            considered during deserialization.
            None by default (no custom layer)
        """
        return None

    @abstractmethod
    def _build_network(self):
        """Abstract method that must be implemented by subclasses to actually
        build the Neural Network graph of layers. This method must return a new
        keras.models.Model object."""

    # ==== Overriding of Utility Methods ====
    #
    def _prepare_metrics_array(self):
        """
        Specialise metrics with extra DNN specific metrics.
        """
        metrics = super(DeepLearningDAP, self)._prepare_metrics_array()

        metrics_shape = (self.iteration_steps, self.feature_steps)
        metrics[self.NN_LOSS] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
        metrics[self.NN_VAL_LOSS] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
        metrics[self.NN_ACC] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
        metrics[self.NN_VAL_ACC] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
        return metrics

    def _compute_extra_step_metrics(self, validation_indices=None, y_true_validation=None,
                                    predictions=None, **extra_metrics):
        """
        Compute extra additional step metrics, specific to Neural Network leaning resulting from
        Keras Models.
        In details, kwargs is expected to contain a key for 'model_history'.

        Parameters
        ----------
        validation_indices: array-like, shape = (n_samples, )
            Indices of validation samples

        y_true_validation: array-like, shape = (n_samples, )
            Array of labels for samples in the validation set

        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_fit_predict` method.

        extra_metrics: dict
            By default, the list of extra metrics will contain model history resulting after training.
            See Also: `_fit_predict` method.
        """

        # Compute Extra Metrics
        model_history = extra_metrics.get(self.HISTORY, None)
        if model_history:
            standard_metrics = ['loss', 'val_loss', 'acc', 'val_acc']
            metric_keys = [self.NN_LOSS, self.NN_VAL_LOSS, self.NN_ACC, self.NN_VAL_ACC]
            for history_key, metric_name in zip(standard_metrics, metric_keys):
                metric_values = model_history.history.get(history_key, None)
                if metric_values:
                    if len(metric_values) < deep_learning_settings.epochs:  # early stopping case
                        values = np.zeros(shape=deep_learning_settings.epochs)
                        values[:len(metric_values)] = metric_values
                    else:
                        values = np.array(metric_values)
                    self.metrics[metric_name][self._iteration_step_nb, self._feature_step_nb] = values

    def _compute_extra_test_metrics(self, y_true_test=None, predictions=None, **extra_metrics):
        """
        Compute extra additional step metrics, specific to Neural Network leaning resulting from
        Keras Models.
        In details, kwargs is expected to contain a key for 'model_history'.

        Parameters
        ----------

        y_true_test: array-like, shape = (n_samples, )
            Array of labels for samples in the validation set

        predictions: array-like, shape = [n_samples] or tuple of array-like objects.
            This parameter contains what has been returned by the `_predict` method.

        extra_metrics: dict
            By default, the list of extra metrics will contain model history resulting after training.
            See Also: `_predict` method.
        """

        # Compute Extra Metrics
        model_history = extra_metrics.get(self.HISTORY, None)
        if model_history:
            standard_metrics = ['loss', 'val_loss', 'acc', 'val_acc']
            metric_keys = [self.NN_LOSS, self.NN_VAL_LOSS, self.NN_ACC, self.NN_VAL_ACC]
            for history_key, metric_name in zip(standard_metrics, metric_keys):
                metric_values = model_history.history.get(history_key, None)
                if metric_values:
                    if len(metric_values) < deep_learning_settings.epochs:  # early stopping case
                        values = np.zeros(shape=deep_learning_settings.epochs)
                        values[:len(metric_values)] = metric_values
                    else:
                        values = np.array(metric_values)
                    self.metrics[self.TEST_SET][metric_name] = values

    def _save_all_metrics_to_file(self, base_output_folder_path, feature_steps, feature_names, sample_names):
        """
        Specialised implementation for Deep learning models, saving to files also
        Network history losses and accuracy.

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

        # Save Base Metrics and CI Metrics as in the Classical DAP
        super(DeepLearningDAP, self)._save_all_metrics_to_file(base_output_folder_path, feature_steps,
                                                               feature_names, sample_names)

        # Save Deep Learning Specific Metrics
        epochs_names = ['epoch {}'.format(e) for e in range(self.learning_epochs)]
        for i_step, step in enumerate(feature_steps):
            for metric_key in self.NETWORK_METRICS:
                self._save_metric_to_file(os.path.join(base_output_folder_path,
                                                       'metric_{}_fs{}.txt'.format(step, metric_key)),
                                          self.metrics[metric_key][:, i_step, :], epochs_names)

    # ==== Overriding of DAP Ops Methods ====
    #
    def _fit_predict(self, model, X_train, y_train, X_validation, y_validation=None):
        """
        Core method to generate metrics on (feature-step) data by fitting
        the input deep learning model and predicting on validation data.
        on validation data.

        Note: The main difference in the implementation of this method relies on the
        fact that the `_fit` method is provided also with validation data (and targets)
        to be fed into the actual `model.fit` method.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

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

        extra_metrics: dict
            List of extra metrics to log during execution. By default, the
            network history will be returned.

        See Also
        --------
        DeepLearningDAP._fit(...)
        """

        # Prepare Data
        X_train = self._prepare_data(X_train, training_data=True)
        y_train = self._prepare_targets(y_train, training_labels=True)
        X_validation = self._prepare_data(X_validation, training_data=False)
        y_validation = self._prepare_targets(y_validation, training_labels=False)

        model, extra_metrics = self._fit(model, X_train, y_train,
                                         X_validation=X_validation, y_validation=y_validation)
        predictions = self._predict(model, X_validation)
        return predictions, extra_metrics

    def _fit(self, model, X_train, y_train, X_validation=None, y_validation=None):
        """
        Default implementation of the training (`fit`) step for an input Keras
        Deep Learning Network model.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

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
        model: the model along with learned weights resulting from the actual training
            process.

        extra_metrics: dict
            List of extra metrics to be logged during execution. Default: `model_history`
        """

        # Setup extra fit parameters from settings
        if X_validation is not None and y_validation is not None:
            self.extra_fit_params['validation_data'] = (X_validation, y_validation)
        else:
            self.extra_fit_params.pop('validation_data', None)

        model_filename = '{}_{}_model.hdf5'.format(self._iteration_step_nb, self._nb_features)
        base_output_folder = self.results_folder
        model_filename = os.path.join(base_output_folder, model_filename)
        callbacks = [ModelCheckpoint(filepath=model_filename, save_best_only=True, save_weights_only=True), ]
        if self.fit_callbacks:
            callbacks.extend(self.fit_callbacks)
            if X_validation is None:
                for callback in callbacks:
                    if hasattr(callback, 'monitor') and callback.monitor == 'val_loss':
                        callback.monitor = 'loss'

        if self.fit_verbose != 0:
            print('Experiment {} - fold {}'.format(self._runstep_nb + 1, self._fold_nb + 1))
            print('Step {}, working with {} features'.format(self._feature_step_nb + 1, self._nb_features))

        model_history = model.fit(X_train, y_train, epochs=self.learning_epochs,
                                  batch_size=self.batch_size, verbose=self.fit_verbose,
                                  callbacks=callbacks, **self.extra_fit_params)
        extra_metrics = {
            self.HISTORY: model_history
        }

        return model, extra_metrics

    def _predict(self, model, X_validation, y_validation=None, **kwargs):
        """
        Default implementation of the inference (`predict`) step of input
        Keras model.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

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

        predicted_class_probs = model.predict(X_validation)
        predicted_classes = predicted_class_probs.argmax(axis=-1)
        return (predicted_classes, predicted_class_probs)

    def run(self, verbose=False):
        dap_model = super(DeepLearningDAP, self).run(verbose)

        if verbose:
            dap_model.summary()

        return dap_model


class DeepLearningDAPRegr(DeepLearningDAP, DAPRegr):
    """Deep Learning DAP Specialisation for Regression Tasks"""

    DAP_REFERENCE_METRIC = DAPRegr.R2_CI
    REF_STEP_METRIC = DAPRegr.R2

    BASE_METRICS = [DAPRegr.EVS, DAPRegr.MAE,
                    DAPRegr.MSE, DAPRegr.MedAE, DAPRegr.R2]

    CI_METRICS = [DAPRegr.EVS_CI, DAPRegr.MAE_CI,
                  DAPRegr.MedAE_CI, DAPRegr.MSE_CI, DAPRegr.R2_CI]

    def __init__(self, experiment):
        DeepLearningDAP.__init__(self, experiment)
        DAPRegr.__init__(self, experiment)

        # ==== Overriding of Utility Methods ====
        #
        def _prepare_metrics_array(self):
            """
            Specialise metrics with extra DNN specific metrics.
            """
            metrics = DAPRegr._prepare_metrics_array(self)

            metrics_shape = (self.iteration_steps, self.feature_steps)
            metrics[self.NN_LOSS] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
            metrics[self.NN_VAL_LOSS] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
            metrics[self.NN_ACC] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
            metrics[self.NN_VAL_ACC] = np.zeros(metrics_shape + (deep_learning_settings.epochs,), dtype=np.float)
            return metrics

        def _compute_step_metrics(self, validation_indices, y_true_validation,
                                  predictions, **extra_metrics):
            DAPRegr._compute_step_metrics(self, validation_indices, y_true_validation, predictions, **extra_metrics)

        def _compute_test_metrics(self, y_true_test, predictions, **extra_metrics):
            DAPRegr._compute_test_metrics(self, y_true_test, predictions, **extra_metrics)


