# This file contains the class to contain a single model version with all its metadata.
# The class contains methods to score, train, and predict the model "model_name".
import datetime
import time


class ModelVersion:
    def __init__(self, parent_set, feature_extractor, model_name, model_hyperparam, module_mgr):
        self.parent_set = parent_set
        self.feature_extractor = feature_extractor  # This is a feature set
        self.module_mgr = module_mgr  # The module manager is stored here as well.
        self.model_name = model_name  # This is a string corresponding to the name of the model
        self.model_class = module_mgr.get_model_class(model_name)  # This is the class of the model (NOT an instance)
        self.model_hyperparam = model_hyperparam  # These are the hyperparameters (dict) used for the model.
        self.trained_last = None  # This is a datetime object (or None) indicating when the model was last trained
        self.scores = {}  # This is a dictionary containing scores obtained by the model e.g. accuracy, f1 score
        self.runtime = None  # This is a float in seconds (or None) indicating how long it took the model to be trained
        self.scoring_runtime = None  # float in seconds (or None) indicating how long it took for model to be scored
        self.to_be_saved = True  # Boolean indicating whether a trained model should be saved together with the project
        self.trained_model = None  # Actual trained model instance
        self.scored = False  # Boolean indicating whether the model version has been scored yet.
        self.user_notes = ""  # This is a string to save user notes for the model version e.g. "This model rocks!"

    # This method scores "model_name" using "scorer_name" with the hyperparameters scorer_hyperparam and
    # model_hyperparam. The resulting scores from the scoring function are stored in self.scores[scorer_name] as a dict.
    def score(self, scorer_name, scorer_hyperparam):
        scorer_func = self.module_mgr.get_scorer_func(scorer_name)

        start_time = time.time()
        scores = scorer_func(self.model_class(**self.model_hyperparam), self.parent_set, self.feature_extractor, **scorer_hyperparam)
        self.scoring_runtime = time.time() - start_time

        self.scored = True
        key = (scorer_name, tuple([value for key, value in sorted(scorer_hyperparam.iteritems())]))
        self.scores[key] = scores

    # This method trains the model with the FULL feature set and stores it in self.trained_model
    def train(self):
        features = self.feature_extractor.get_features_array(self.parent_set)
        labels = self.feature_extractor.get_labels_array(self.parent_set)

        model = self.model_class(**self.model_hyperparam)
        start_time = time.time()
        if model.train(features, labels):
            self.runtime = time.time() - start_time

            self.trained_model = model
            self.trained_last = datetime.datetime.today()
        else:
            print "Training failed"

    # This method will fail if there is no model in self.trained_model
    # outside_set is a ParentSet object that may or may not have a .labels attribute
    def predict(self, outside_set):
        if not self.trained_model:
            print "Model is untrained"
            return None
        else:
            return self.trained_model.predict(self.feature_extractor.get_features_array(outside_set))


if __name__ == '__main__':
    from sklearn import datasets
    import modulemanager, parameterspinner
    import parentset
    import featureextractor
    m = modulemanager.ModuleManager()
    hyperparams_model = parameterspinner.ParameterSpinner.use_default_values(m.get_model_hyperparams("Logistic Regression"))
    print hyperparams_model
    hyperparams_scorer = parameterspinner.ParameterSpinner.use_default_values(m.get_scorer_hyperparams("General Cross Validation"))
    print hyperparams_scorer
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    parent_set = parentset.ParentSet(X, y)
    my_class = ModelVersion(parent_set, featureextractor.FeatureExtractor(), "Logistic Regression", hyperparams_model, m)
    my_class.score("General Cross Validation", hyperparams_scorer)
    print my_class.scores
    print my_class.trained_model
    my_class.train()
    print my_class.trained_model
    print my_class.trained_last
    print my_class.runtime
    print my_class.scoring_runtime
