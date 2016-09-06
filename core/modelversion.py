# This file contains the class to contain a single model version with all its metadata.
# The class contains methods to score, train, and predict the model "model_name".
import datetime


class ModelVersion:
    def __init__(self, feature_set, labels, scorer_name, model_name, scorer_hyperparam, model_hyperparam, module_mgr):
        self.feature_set = feature_set  # This is a feature set
        self.labels = labels  # Corresponding correct labels for feature set
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
        self.scoring_runtime = None  # int in seconds (or None) indicating how long it took for model to be scored
        self.to_be_saved = True  # Boolean indicating whether a trained model should be saved together with the project
        self.trained_model = None  # Actual trained model instance
        self.user_notes = ""  # This is a string to save user notes for the model version e.g. "This model rocks!"

    # This method scores "model_name" using "scorer_name" with the hyperparameters scorer_hyperparam and
    # model_hyperparam. The resulting scores from the scoring function are stored in scores.
    def score(self):
        model = self.module_mgr.get_model(self.model_name, self.model_hyperparam)
        scorer_function = self.module_mgr.get_scorer_func(self.scorer_name)
        scores = scorer_function(model, self.feature_set, self.labels, **self.scorer_hyperparam)
        self.scores = scores

    # This method trains the model with the FULL feature set and stores it in self.trained_model
    def train(self):
        model = self.module_mgr.get_model(self.model_name, self.model_hyperparam)
        if model.train(self.feature_set, self.labels):
            self.trained_model = model
            self.trained_last = datetime.datetime.today()
        else:
            print "Training failed"

    # This method will fail if there is no model in self.trained_model
    def predict(self, feature_set):
        if not self.trained_model:
            print "Model is untrained"
            return None
        else:
            return self.trained_model.predict(feature_set)


if __name__ == '__main__':
    pass
