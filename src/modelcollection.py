# This module contains the class for easily creating and managing differently trained versions of the same model.
import modelversion
import parameterspinner


class ModelCollection:
    def __init__(self, model_name, module_mgr):
        self.model_name = model_name  # This is a string corresponding to the name of the model
        self.module_mgr = module_mgr  # This contains the module manager used to interface with the various models and
        # scorers available to Ensemble Helper.
        self.model_versions = {}  # This dictionary contains the model versions in the model collection.

    # model_version is a ModelVersion object.
    # This method adds model_version into the dictionary self.model_versions if it is not already there.
    # If it is already in the collection, it does nothing.
    def _add_model_version_to_collection(self, model_version):
        dict_key = [model_version.scorer_name]
        dict_key.extend([value for key, value in sorted(model_version.model_hyperparam.iteritems())])
        dict_key.extend([value for key, value in sorted(model_version.scorer_hyperparam.iteritems())])
        dict_key = tuple(dict_key)
        if dict_key in self.model_versions:
            pass
        else:
            self.model_versions[dict_key] = model_version

    # This method generates a valid model based on the given feature set, scorer, and hyperparameters (model or scorer)
    # and adds it to self.model_versions if it doesn't already exist there.
    def generate_and_add_model(self, feature_set, labels, scorer_name, hyperparams_scorer, hyperparams_model):
        new_model = modelversion.ModelVersion(feature_set, labels, scorer_name, self.model_name, hyperparams_scorer, hyperparams_model, self.module_mgr)
        self._add_model_version_to_collection(new_model)


if __name__ == "__main__":
    import modulemanager, parameterspinner
    from sklearn import datasets
    m = modulemanager.ModuleManager()
    hyperparams_model = parameterspinner.ParameterSpinner.use_default_values(m.get_model_hyperparams("Logistic Regression"))
    hyperparams_scorer = parameterspinner.ParameterSpinner.use_default_values(m.get_scorer_hyperparams("General Cross Validation"))
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    my_collection = ModelCollection("Logistic Regression", m)
    my_collection.generate_and_add_model(X, y, "General Cross Validation", hyperparams_scorer, hyperparams_model)
    print my_collection.model_versions