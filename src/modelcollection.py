# This module contains the class for easily creating and managing differently trained versions of the same model.
import modelversion
import parameterspinner
import multiprocessing


class ModelCollection:
    def __init__(self, model_name, parent_set, module_mgr):
        self.model_name = model_name  # This is a string corresponding to the name of the model
        self.parent_set = parent_set  # This is a ParentSet instance corresponding to the ParentSet used to score and
        # train the models in the ModelCollection object.
        self.module_mgr = module_mgr  # This contains the module manager used to interface with the various models and
        # scorers available to Ensemble Helper.
        self.model_versions = {}  # This dictionary contains the model versions in the model collection.

    # model_version is a ModelVersion object.
    # This method adds model_version into the dictionary self.model_versions if it is not already there.
    # If it is already in the collection, it does nothing.
    def _add_model_version_to_collection(self, model_version):
        dict_key = [model_version.feature_extractor]
        dict_key.extend([value for key, value in sorted(model_version.model_hyperparam.iteritems())])
        dict_key = tuple(dict_key)
        if dict_key in self.model_versions:
            # print "already here!", dict_key
            pass
        else:
            self.model_versions[dict_key] = model_version

    # This method generates a valid model based on the given feature set, scorer, and hyperparameters (model or scorer)
    # and adds it to self.model_versions if it doesn't already exist there.
    def generate_and_add_model(self, feature_extractor, model_hyperparam):
        new_model = modelversion.ModelVersion(self.parent_set, feature_extractor, self.model_name, model_hyperparam, self.module_mgr)
        self._add_model_version_to_collection(new_model)

    # This method uses param_grid to automatically generate and add models that are the result of the combination of
    # the parameters from param_grid. Note that this method assumes that the feature set
    # passed to this method does not change throughout the exhaustive grid search. Therefore, this method
    # is a grid search only changing the model hyperparameters.
    def generate_models_from_grid_hyperparam(self, feature_extractor, param_grid):
        hyperparam_info_dict = self.module_mgr.get_model_hyperparams(self.model_name)
        for model_hyperparam in parameterspinner.ParameterSpinner.exhaustive_search_iterator(hyperparam_info_dict, param_grid):
            self.generate_and_add_model(feature_extractor, model_hyperparam)

    # This method calculates a score for all unscored model versions in self.model_versions using score_name and the
    # corresponding scorer_hyperparam.
    # Support for parallel processing still in the works
    def score_all_models(self, scorer_name, scorer_hyperparam):
        for key, model_version in sorted(self.model_versions.iteritems()):
            if not model_version.scored:
                try:
                    model_version.score(scorer_name, scorer_hyperparam)
                    # print model_version.scores["accuracy"], model_version.model_hyperparam["C"], model_version.model_hyperparam["penalty"]
                except Exception as e:
                    print e

    # Experimental method for calculating the scores of all the models in self.model_versions in parallel
    def score_all_models_parallel(self, scorer_name, scorer_hyperparam):
        my_pool = multiprocessing.Pool()
        iterable = [(model_version, scorer_name, scorer_hyperparam) for key, model_version in sorted(self.model_versions.iteritems())]
        new_model_versions = my_pool.map(score_parallel, iterable)
        for new_model, key in zip(new_model_versions, sorted(self.model_versions)):
            if new_model is not None:
                self.model_versions[key] = new_model


# __packed_arguments is a tuple in the form of
# (<modelversion instance> modelversionpassed, <string> scorer_name, <dict> scorer_hyperparam)
# It is unpacked in the first line of the function
def score_parallel(__packed_arguments):
    modelversionpassed, scorer_name, scorer_hyperparam = __packed_arguments
    if not modelversionpassed.scored:
        try:
            modelversionpassed.score(scorer_name, scorer_hyperparam)
        except Exception as e:
            print e
            return None
    return modelversionpassed


if __name__ == "__main__":
    import modulemanager
    from sklearn import datasets
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
    for key, value in sorted(my_collection.model_versions.iteritems()):
        print key, ":", value
    my_collection.score_all_models_parallel("General Cross Validation", hyperparams_scorer)
    for key, model_version in sorted(my_collection.model_versions.iteritems()):
        print model_version.scores, model_version.model_hyperparam["C"], model_version.model_hyperparam["penalty"], model_version.scoring_runtime

