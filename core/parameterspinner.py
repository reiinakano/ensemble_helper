# This module contains the class for a parameter spinner, containing different methods to automatically generate valid
# hyperparameters from a hyperparameter information dictionary i.e. dictionary "hyperparam" in a hyperparam.py file,


class ParameterSpinner:
    def __init__(self, hyperdict):
        self.hyperdict = hyperdict

    # Returns the list of hyperparameter names to the dictionary
    def get_parameter_list(self):
        return [key for key in self.hyperdict]

