# This file contains the interface for interacting with the models and scorers in the packages enModels and enScorers
import os, importlib


class ModuleManager:
    # To initialize ModuleManager, it must crawl through the directories enModels and enScorers to determine which
    # models and scoring methods are correctly configured and available for use.
    def __init__(self):
        self.available_models, self.available_scorers = self._get_available_models(), self._get_available_scorers()

    # This function should only be used while initializing the module manager class.
    # It parses the enModels folder, looks for correctly configured models, and adds them to the available_models
    # dictionary. Each entry in the dictionary is accessed by the model name. The corresponding value is another
    # dictionary containing the particular model's class and hyperparameters.
    def _get_available_models(self):
        available_models = {}
        for dir_name in [f for f in os.listdir('./enModels') if os.path.isdir('./enModels/' + f)]:
            # insert instructions to test if path is a legitimate model package. If it is, add it to available_models.
            imported_module = importlib.import_module('enModels.' + dir_name + '.modelclass')  # Import module
            model_class = imported_module.ModelClass  # get class ModelClass from imported module
            model_name = model_class.model_name()  # get name of model from ModelClass class's static method model_name()
            imported_module = importlib.import_module('enModels.' + dir_name + '.modelhyperparam')  # Import hyperparam module
            model_hyperparam = imported_module.hyperparam  # Get model hyperparams
            available_models[model_name] = {}  # add model name to dictionary. Initialize with another dictionary.
            available_models[model_name]["hyperparam"] = model_hyperparam  # store hyperparams of model in dictionary.
            available_models[model_name]["class"] = model_class  # store class in model name
        return available_models

    # This function should only be used while initializing the module manager class.
    # It parses the enScorers folder, looks for correctly configured scorers, and adds them to the available_scorers
    # dictionary. Each entry in the dictionary is accessed by the scorer name. The corresponding value is another
    # dictionary containing the particular scorer's score function and hyperparameters.
    def _get_available_scorers(self):
        available_scorers = {}
        for dir_name in [f for f in os.listdir('./enScorers') if os.path.isdir('./enScorers/' + f)]:
            # Insert instruction to test if path is a legitimate scorer package. If it is, add it to available models.
            imported_module = importlib.import_module('enScorers.' + dir_name + '.scorer')  # Import module
            scorer_function = imported_module.score
            scorer_name = imported_module.scorer_name()
            imported_module = importlib.import_module('enScorers.' + dir_name + '.scorerhyperparam')  # Import scorer hyperparam module
            scorer_hyperparam = imported_module.hyperparam
            available_scorers[scorer_name] = {}
            available_scorers[scorer_name]["score"] = scorer_function
            available_scorers[scorer_name]["hyperparam"] = scorer_hyperparam
        return available_scorers

    # Returns instance of model with name "model_name" and initialized with dictionary "hyperparams"
    def get_model(self, model_name, hyperparams):
        try:
            model = self.available_models[model_name]["class"](**hyperparams)
        except KeyError:
            print "Model " + model_name + " does not exist."
            return None
        return model

    # Returns model hyperparameters dictionary of model_name
    def get_model_hyperparams(self, model_name):
        try:
            hyperparam = self.available_models[model_name]["hyperparam"]
        except KeyError:
            print "Model " + model_name + " does not exist."
            return None
        return hyperparam

    def get_scorer_func(self, scorer_name):
        try:
            scorer = self.available_scorers[scorer_name]["score"]
        except KeyError:
            print "Scorer " + scorer_name + " does not exist."
            return None
        return scorer


if __name__ == "__main__":
    print ['./enScorers/' + f for f in os.listdir('./enScorers') if os.path.isdir('./enScorers/' + f)]
    print ['./enModels/' + f for f in os.listdir('./enModels') if os.path.isdir('./enModels/' + f)]
    m = ModuleManager()
    for key, value in m.available_models.iteritems():
        print key
        print value["class"]
        print value["hyperparam"]
    print ""
    for key, value in m.available_scorers.iteritems():
        print key
        print value["score"]
        print value["hyperparam"]
    hyperparams = {}
    for key, value in m.get_model_hyperparams("Logistic Regression").iteritems():
        hyperparams[key] = value["default"]
    print m.get_model("Logistic Regression", hyperparams)