# This file contains the class to contain a single model version and its metadata.


class ModelVersion:
    def __init__(self, feature_set, scorer_name, model_name, scorer_hyperparam, model_hyperparam, module_mgr):
        self.feature_set = feature_set  # This is a feature set object
        self.scorer_name = scorer_name  # This is a string corresponding to the name of the scorer
        self.model_name = model_name  # This is a string corresponding to the name of the model
        self.scorer_hyperparam = scorer_hyperparam  # These are the hyperparameters (dict) used for the scorer.
        self.model_hyperparam = model_hyperparam  # These are the hyperparameters (dict) used for the model.
        # This is the module manager passed to ModelVersion and is used for interacting
        # with the particular model and scorer indicated by model_name and scorer_name.
        self.module_mgr = module_mgr
        self.trained_last = None  # This is a datetime object (or None) indicating when the model was last trained
        self.scores = {}  # This is a dictionary containing scores obtained by the model e.g. accuracy, f1 score
        self.runtime = None  # This is a int in seconds (or None) indicating how long it took the model to be trained
        self.to_be_saved = True  # Boolean indicating whether a trained model should be saved together with the project
        self.actual_model = None  # Actual model instance

    def train(self):
        model = self.module_mgr.get_model(self.model_name, self.model_hyperparam)
        scorer_function = self.module_mgr.get_scorer_func(self.scorer_name)
