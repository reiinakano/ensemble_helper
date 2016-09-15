# This module contains the class for implementing a hard voting (uses majority classification, not average voting)
# ensemble.
import parentset
import featureextractor


class HardVotingClassifier:
    def __init__(self, parent_set):
        self.parent_set = parent_set
        self.scores = {}  # scores dictionary
        self.basemodelversions = {}  # ModelVersions are keys to the dictionary. Values are the weight.
        self.user_notes = ""
        self.scored = False
        self.trained_model = False  # Unlike in ModelVersion instances, trained_model in ensembles are booleans.

    def add_model(self, modelversion, weight=1):
        self.basemodelversions[modelversion] = weight

    def score(self, scorer_name, scorer_hyperparam, module_mgr):
        features = featureextractor.FeatureExtractor().get_features_array(self.parent_set)
        labels = featureextractor.FeatureExtractor().get_labels_array(self.parent_set)
        scorer_func = module_mgr.get_scorer_func(scorer_name)

        scores = scorer_func(ForScorer(self), features, labels, **scorer_hyperparam)

        self.scored = True
        key = (scorer_name, tuple([value for key, value in sorted(scorer_hyperparam.iteritems())]))
        self.scores[key] = scores

    def train(self):
        for modelversion, weight in sorted(self.basemodelversions.iteritems()):
            if not modelversion.trained_model:
                modelversion.train()
        self.trained_model = True

    def predict(self, outside_set):
        for modelversion, weight in sorted(self.basemodelversions.iteritems()):
            if not modelversion.trained_model:
                print "Not all models in ensemble are trained."
                return None
        classifications = [modelversion.predict(outside_set) for modelversion in sorted(self.basemodelversions)]
        weights = [value for key, value in sorted(self.basemodelversions.iteritems())]
        classifications = get_majority(classifications, weights)
        return classifications


# This class is used to play nice with the syntax of the scorers.
# This is probably the ugliest part of the code. I would love to change this someday.
class ForScorer:
    def __init__(self, hardvotingclassifier):
        self.models = []
        self.feature_extractors = []
        self.weights = []
        for modelversion, weight in sorted(hardvotingclassifier.basemodelversions.iteritems()):
            self.models.append(modelversion.model_class(**modelversion.model_hyperparam))
            self.feature_extractors.append(modelversion.feature_extractor)
            self.weights.append(weight)

    def train(self, feature_set, labels):
        parent_set = parentset.ParentSet(feature_set, labels)
        for model, feature_extractor in zip(self.models, self.feature_extractors):
            model.train(feature_extractor.get_features_array(parent_set), feature_extractor.get_labels_array(parent_set))

    def predict(self, feature_set_to_predict):
        parent_set = parentset.ParentSet(feature_set_to_predict, None)
        classifications = []
        for model, feature_extractor in zip(self.models, self.feature_extractors):
            classifications.append(model.predict(feature_extractor.get_features_array(parent_set)))
        classifications = get_majority(classifications, self.weights)
        return classifications


# Sample use case:
# classifications = [["r","y","r","r"],["y","r","r","r"],["y", "y", "y", "y",],["y","y","r","r"]]
# weights = [1, 1, 1, 1]
# get_majority(classifications, weights) returns ["y","y","r","r"]
def get_majority(classifications, weights):
    new_classifications = []
    for votes in zip(*classifications):
        my_dict = {}
        for index, vote in enumerate(votes):
            if vote not in my_dict:
                my_dict[vote] = weights[index]
            else:
                my_dict[vote] += weights[index]
        new_classifications.append(max(my_dict, key=my_dict.get))
    return new_classifications


if __name__ == "__main__":
    print get_majority([["r","y","r","r"],
                        ["y","r","r","r"],
                        ["y","y","y","y"],
                        ["y","y","r","r"]],
                       [1, 1, 1, 1])
    import modulemanager
    from sklearn import datasets
    import parameterspinner
    from modelcollection import ModelCollection
    m = modulemanager.ModuleManager()
    hyperparams_model = parameterspinner.ParameterSpinner.use_default_values(m.get_model_hyperparams("Logistic Regression"))
    hyperparams_scorer = parameterspinner.ParameterSpinner.use_default_values(m.get_scorer_hyperparams("General Cross Validation"))
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    parent_set = parentset.ParentSet(X, y)
    my_collection = ModelCollection("Logistic Regression", parent_set, m)
    param_grid = [{"C": [0.01, 0.1, 1.0, 10.0, 100.0], "n_jobs": [1, -1]},
                  {"C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], "penalty": ["l1", "l2"]}]
    my_collection.generate_models_from_grid_hyperparam(featureextractor.FeatureExtractor(), param_grid)
    my_collection.score_all_models_parallel("General Cross Validation", hyperparams_scorer)
    hvc = HardVotingClassifier(parent_set)
    for key, model_version in sorted(my_collection.model_versions.iteritems()):
        print model_version.scores
    for key, value in sorted(my_collection.model_versions.iteritems()):
        if value.scores[("General Cross Validation", (3, True, True, True, True, True, 'stratified', False))]["accuracy"] > 0.96:
            hvc.add_model(value)
    print hvc.basemodelversions
    hvc.score("General Cross Validation", hyperparams_scorer, m)
    print hvc.scores
