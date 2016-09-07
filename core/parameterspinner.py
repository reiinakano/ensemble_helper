# This module contains the class for a parameter spinner, containing different methods to automatically generate valid
# hyperparameters from a hyperparameter information dictionary i.e. dictionary "hyperparam" in a hyperparam.py file,


class ParameterSpinner:
    def __init__(self, hyperdict=None):
        self.hyperdict = hyperdict

    # Returns the list of hyperparameter names to the dictionary
    @staticmethod
    def get_parameter_list(hyperdict):
        return [key for key in hyperdict]

    # Returns valid hyperparameters using only the default values of the dictionary
    @staticmethod
    def use_default_values(hyperdict):
        return {key: value["default"] for key, value in hyperdict.iteritems()}


if __name__ == "__main__":
    import modulemanager
    m = modulemanager.ModuleManager()
    hyperdictt = m.get_model_hyperparams("Logistic Regression")
    print ParameterSpinner.use_default_values(hyperdictt)
