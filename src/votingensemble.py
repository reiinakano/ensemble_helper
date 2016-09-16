# This module contains the class for organizing a voting ensemble (based on votingclassifier.VotingClassifier) from
# established model versions.
from enEnsemble import votingclassifier


class VotingEnsemble:
    def __init__(self, parent_set):
        self.parent_set = parent_set
        self.scores = {}  # scores dictionary
        self.basemodelversions = {}  # ModelVersions are keys to the dictionary. Values are the weight.
        self.user_notes = ""
        self.scored = False
        self.trained_model = False  # Unlike in ModelVersion instances, trained_model in ensembles are booleans.

    def add_model(self, modelversion, weight=1):
        self.basemodelversions[modelversion] = weight

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
        classifications = votingclassifier.VotingClassifier.get_majority(classifications, weights)
        return classifications

    def score(self, scorer_name, scorer_hyperparam, module_mgr):
        scorer_func = module_mgr.get_scorer_func(scorer_name)

        scores = scorer_func(self.return_model(), self.parent_set.features, self.parent_set.labels, **scorer_hyperparam)

        self.scored = True
        key = (scorer_name, tuple([value for key, value in sorted(scorer_hyperparam.iteritems())]))
        self.scores[key] = scores

    # Returns instance of estimator (votingclassifier.VotingClassifier instance)
    def return_model(self):
        basemodels = []
        weights = []
        for modelversion, weight in sorted(self.basemodelversions.iteritems()):
            basemodels.append(modelversion.return_model())
            weights.append(weight)
        return votingclassifier.VotingClassifier(basemodels, weights)


if __name__ == "__main__":
    import modulemanager
    from sklearn import datasets
    import parameterspinner
    from modelcollection import ModelCollection
    import parentset
    import featureextractor
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
    hvc = VotingEnsemble(parent_set)
    for key, model_version in sorted(my_collection.model_versions.iteritems()):
        print model_version.scores
    for key, value in sorted(my_collection.model_versions.iteritems()):
        if value.scores[("General Cross Validation", (3, True, True, True, True, True, 'stratified', False))]["accuracy"] > 0.9:
            hvc.add_model(value)
    print hvc.basemodelversions
    hvc.score("General Cross Validation", hyperparams_scorer, m)
    print hvc.scores
