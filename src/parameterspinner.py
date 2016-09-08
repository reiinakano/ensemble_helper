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
        for argument_dict in ParameterGrid(param_grid):
            default = ParameterSpinner.use_default_values(hyperdict)
            for key, value in argument_dict.iteritems():
                if key not in default:
                    raise KeyError("Parameter '{}' not in dictionary.".format(key))
                default[key] = value
            yield default


if __name__ == "__main__":
    import modulemanager
    m = modulemanager.ModuleManager()
    hyperdictt = m.get_model_hyperparams("Logistic Regression")
    print "default: ", ParameterSpinner.use_default_values(hyperdictt)
    param_grid = [{"C": [2, 3], "n_jobs": [2, 1, 3, -1]},
                  {"C": [2, 4], "penalty": ["l1", "l2"]}]
    for my_dict in ParameterSpinner.exhaustive_search_iterator(hyperdictt, param_grid):
        print my_dict