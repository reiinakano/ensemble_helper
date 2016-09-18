# This module contains the class for organizing a stacked ensemble (based on stackingclassifier.StackingClassifier) from
# established model versions.
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from enEnsemble import stackingclassifier


class StackedEnsemble:
    def __init__(self, parent_set, secondary_model=None, n_folds=2, use_features_in_secondary=False,
                 use_predict_proba=False):
        self.parent_set = parent_set
        self.secondary_model = secondary_model
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.use_predict_proba = use_predict_proba
        self.scores = {}  # scores dictionary
        self.basemodelversions = set([])  # set of ModelVersions
        self.user_notes = ""
        self.scored = False
        self.trained_model = False  # Unlike in ModelVersion instances, trained_model in ensembles are booleans.

    def add_model(self, modelversion):
        self.basemodelversions.add(modelversion)
        self.trained_model = False

    def add_secondary_model(self, model):
        self.secondary_model = model
        self.trained_model = False  # You will need to train this new secondary model again.

    def train(self):
        for modelversion in sorted(self.basemodelversions):
            if not modelversion.trained_model:
                modelversion.train()

        if self.trained_model:
            return None

        if self.secondary_model is None:
            print "No secondary model assigned yet!"
            return None

        skf = StratifiedKFold(self.parent_set.labels, n_folds=self.n_folds)
        all_model_predictions = np.array([]).reshape(len(self.parent_set.labels), 0)
        for model in [basemodel.return_model() for basemodel in sorted(self.basemodelversions)]:
            if not self.use_predict_proba:
                single_model_prediction = np.array([]).reshape(0, 1)
            else:
                single_model_prediction = np.array([]).reshape(0, len(set(self.parent_set.labels)))

            for train_index, test_index in skf:
                if not self.use_predict_proba:
                    prediction = model.fit(self.parent_set.features[train_index], self.parent_set.labels[train_index]).predict(self.parent_set.features[test_index])
                    prediction = prediction.reshape(prediction.shape[0], 1)
                else:
                    prediction = model.fit(self.parent_set.features[train_index], self.parent_set.labels[train_index]).predict_proba(self.parent_set.features[test_index])
                single_model_prediction = np.vstack([single_model_prediction.astype(prediction.dtype), prediction])

            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))

        # We have to shuffle the labels in the same order as we generated predictions during CV
        # (we kinda shuffled them when we did Stratified CV)
        # We also do the same with the features (we will need this only IF use_features_in_secondary is True)
        reordered_labels = np.array([]).astype(self.parent_set.labels.dtype)
        reordered_features = np.array([]).reshape((0, self.parent_set.features.shape[1])).astype(self.parent_set.features.dtype)
        for train_index, test_index in skf:
            reordered_labels = np.concatenate((reordered_labels, self.parent_set.labels[test_index]))
            reordered_features = np.concatenate((reordered_features, self.parent_set.features[test_index]))

        # Fit the secondary model
        if not self.use_features_in_secondary:
            self.secondary_model.fit(all_model_predictions, reordered_labels)
        else:
            self.secondary_model.fit(np.hstack((reordered_features, all_model_predictions)), reordered_labels)

        self.trained_model = True

    def predict(self, outside_set):
        if not self.trained_model:
            print "Ensemble not fully trained"
            return None

        all_model_predictions = np.array([]).reshape(len(outside_set), 0)
        for model in sorted(self.basemodelversions):
            if not self.use_predict_proba:
                single_model_prediction = model.predict(outside_set)
                single_model_prediction = single_model_prediction.reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(outside_set)
            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))
        if not self.use_features_in_secondary:
            return self.secondary_model.predict(all_model_predictions)
        else:
            return self.secondary_model.predict(np.hstack((outside_set, all_model_predictions)))

    def score(self, scorer_name, scorer_hyperparam, module_mgr):
        scorer_func = module_mgr.get_scorer_func(scorer_name)

        scores = scorer_func(self.return_model(), self.parent_set.features, self.parent_set.labels, **scorer_hyperparam)

        self.scored = True
        key = (scorer_name, tuple([value for key, value in sorted(scorer_hyperparam.iteritems())]))
        self.scores[key] = scores

    # Returns instance of estimator (stackingclassifier.StackingClassifier instance)
    def return_model(self):
        if self.secondary_model is None:
            print "No secondary model is chosen yet."
            return None
        basemodels = [modelversion.return_model() for modelversion in sorted(self.basemodelversions)]
        return stackingclassifier.StackingClassifier(basemodels, self.secondary_model, n_folds=self.n_folds,
                                                     use_features_in_secondary=self.use_features_in_secondary,
                                                     use_predict_proba=self.use_predict_proba)

if __name__ == "__main__":
    import modulemanager
    from sklearn import datasets
    import parameterspinner
    from modelcollection import ModelCollection
    import parentset
    import featureextractor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    m = modulemanager.ModuleManager()
    hyperparams_model = parameterspinner.ParameterSpinner.use_default_values(m.get_model_hyperparams("Logistic Regression"))
    hyperparams_scorer = parameterspinner.ParameterSpinner.change_default(m.get_scorer_hyperparams("General Cross Validation"), {"N": 10})
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    parent_set = parentset.ParentSet(X, y)
    my_collection = ModelCollection("Logistic Regression", parent_set, m)
    param_grid = [{"C": [0.01, 0.1, 1.0, 10.0, 100.0], "n_jobs": [1, -1]},
                  {"C": [0.00001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], "penalty": ["l1", "l2"]}]
    my_collection.generate_models_from_grid_hyperparam(featureextractor.FeatureExtractor(), param_grid)
    my_collection.score_all_models_parallel("General Cross Validation", hyperparams_scorer)
    hvc = StackedEnsemble(parent_set, m.get_model("Logistic Regression", hyperparams_model), 2, False, True)
    for key, model_version in sorted(my_collection.model_versions.iteritems()):
        print model_version.scores
    for key, value in sorted(my_collection.model_versions.iteritems()):
        if value.scores[("General Cross Validation", (10, True, True, True, True, True, 'stratified', False))]["accuracy"] > 0.96:
            hvc.add_model(value)
    print hvc.basemodelversions
    hvc.score("General Cross Validation", hyperparams_scorer, m)
    print hvc.scores
    hvc.train()
    print hvc.predict(X)
    print y
