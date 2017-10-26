"""Implements some pre-defined base runners to be used in DAP with sklearn models."""

from dap import DAP, DAPRegr


class BaseRunner:

    def __init__(self, **hyper_params):
        self._hyper_params = hyper_params

    @property
    def hyper_params(self):
        return self._hyper_params

    @hyper_params.setter
    def hyper_params(self, **params):
        self._hyper_params = params


# =========================================================================
# Random Forest Runner
# =========================================================================

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestRunnerDAP(DAP, BaseRunner):
    """Specialisation of the DAP for RandomForest Classifier"""

    def __init__(self, experiment):
        DAP.__init__(self, experiment=experiment)
        # Initialise Hyper parameters with RandomForestClassifier's default ones
        BaseRunner.__init__(self, **RandomForestClassifier().get_params())

    # ==== DAP Abstract Methods Implementations ====

    @property
    def ml_model_name(self):
        return 'random_forest'

    def create_ml_model(self):
        """Instantiate a new Random Forest model to be used in the fit-predict step."""
        return RandomForestClassifier(**self.hyper_params)


class RandomForestRegressorRunnerDAP(DAPRegr, BaseRunner):
    """DAP Specialisation for Random Forest Regressor"""

    def __init__(self, experiment):
        DAP.__init__(self, experiment=experiment)
        # Initialise Hyper parameters with RandomForestRegressor's default ones
        BaseRunner.__init__(self, **RandomForestRegressor().get_params())

    @property
    def ml_model_name(self):
        return 'random_forest_regressor'

    def create_ml_model(self):
        """"""
        return RandomForestRegressor(**self.hyper_params)


# =========================================================================
# Support Vector Machines Runner
# =========================================================================

from sklearn.svm import SVC, SVR


class SupportVectorRunnerDAP(DAP, BaseRunner):
    """Specialisation of the DAP for Support Vector Classifier"""

    def __init__(self, experiment):
        DAP.__init__(self, experiment=experiment)
        # Initialise Hyper parameters with SupportVectorClassifier's default ones
        BaseRunner.__init__(self, **SVC().get_params())


    # ==== Abstract Methods Implementation ====

    @property
    def ml_model_name(self):
        return 'svm'

    def create_ml_model(self):
        """Instantiate a new SVC model to be used in the fit-predict step."""
        return SVC(**self._hyper_params)


class SupportVectorRegressorRunnerDAP(DAPRegr, BaseRunner):
    """DAP Specialisation for Support Vector Regressor"""

    def __init__(self, experiment):
        DAP.__init__(self, experiment=experiment)
        # Initialise Hyper parameters with SupportVectorClassifier's default ones
        BaseRunner.__init__(self, **SVR().get_params())

    @property
    def ml_model_name(self):
        return 'svr'

    def create_ml_model(self):
        """Instantiate a new SVR model to be used in the fit-predict step."""
        return SVR(**self.hyper_params)