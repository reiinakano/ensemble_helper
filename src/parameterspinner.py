# This module contains the class for a parameter spinner, containing different methods to automatically generate valid
# hyperparameters from a hyperparameter information dictionary i.e. dictionary "hyperparam" in a hyperparam.py file,
from sklearn.grid_search import ParameterGrid
from collections import Mapping


class ParameterSpinner:
    def __init__(self, hyperdict):
        self.hyperdict = hyperdict

    # Returns the list of hyperparameter names to the dictionary
    @staticmethod
    def get_parameter_list(hyperdict):
        return [key for key in hyperdict]

    # Returns valid hyperparameters using only the default values of the dictionary
    @staticmethod
    def use_default_values(hyperdict):
        return {key: value["default"] for key, value in hyperdict.iteritems()}

    # This function is pretty much the exhaustive grid search using sklearn's GridSearchCV.
    # "params" is a list of dictionaries representing various grids to be generated.
    # The function returns an iterator that yields a valid hyperparam dictionary per iteration.
    @staticmethod
    def exhaustive_search_iterator(hyperdict, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            pass


if __name__ == "__main__":
    import modulemanager
    m = modulemanager.ModuleManager()
    hyperdictt = m.get_model_hyperparams("Logistic Regression")
    print ParameterSpinner.use_default_values(hyperdictt)
    for g in ParameterGrid([{"d":[2,3], "w":[2,1,3,3]},{"C":[2, 4], "f":[2, "str"]}]):
        print g