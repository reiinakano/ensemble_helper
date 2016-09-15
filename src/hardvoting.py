# This module contains the class for implementing a hard voting (uses majority classification, not average voting)
# ensemble.
import parentset
import featureextractor

class HardVotingClassifier:
    def __init__(self, parent_set):
        self.parent_set = parent_set
        self.scores = {}  # scores dictionary
        self._modelversions = set([])  # contains the ModelVersions that are in the HardVotingClassifier
        self.user_notes = ""
        self.scored = False

    def add_model(self, modelversion):
        self._modelversions.add(modelversion)

    def score(self, scorer_name, scorer_hyperparam, module_mgr):
        features = featureextractor.FeatureExtractor().get_features_array(self.parent_set)
        labels = featureextractor.FeatureExtractor().get_labels_array(self.parent_set)
        scorer_func = module_mgr.get_scorer_func(scorer_name)

        scores = scorer_func(ForScorer(self), features, labels, **scorer_hyperparam)

        self.scored = True
        key = (scorer_name, tuple([value for key, value in sorted(scorer_hyperparam.iteritems())]))
        self.scores[key] = scores

    def train(self):
        for modelversion in sorted(self._modelversions):
            if not modelversion.trained_model:
                modelversion.train()

    def predict(self, outside_set):
        for modelversion in sorted(self._modelversions):
            if not modelversion.trained_model:
                print "Not all models in ensemble are trained."
                return None
        classifications = [modelversion.predict(outside_set) for modelversion in sorted(self._modelversions)]
        classifications = get_majority(classifications)
        return classifications


# This class is used to play nice with the syntax of the scorers.
class ForScorer:
    def __init__(self, hardvotingclassifier):
        self.models = []
        self.feature_extractors = []
        for modelversion in sorted(hardvotingclassifier._modelversions):
            self.models.append(modelversion.model_class(**modelversion.model_hyperparam))
            self.feature_extractors.append(modelversion.feature_extractor)

    def train(self, feature_set, labels):
        parent_set = parentset.ParentSet(feature_set, labels)
        for model, feature_extractor in zip(self.models, self.feature_extractors):
            model.train(feature_extractor.get_features_array(parent_set), feature_extractor.get_labels_array(parent_set))

    def predict(self, feature_set_to_predict):
        parent_set = parentset.ParentSet(feature_set_to_predict, None)
        classifications = []
        for model, feature_extractor in zip(self.models, self.feature_extractors):
            classifications.append(model.predict(feature_extractor.get_features_array(parent_set)))
        classifications = get_majority(classifications)
        return classifications


# Sample use case:
# classifications = [["r","y","r","r"],["y","r","r","r"],["y", "y", "y", "y",],["y","y","r","r"]]
# get_majority(classifications) returns ["y","y","r","r"]
def get_majority(classifications):
    new_classifications = []
    for votes in zip(*classifications):
        my_dict = {}
        for vote in votes:
            if vote not in my_dict:
                my_dict[vote] = 1
            else:
                my_dict[vote] += 1
        new_classifications.append(max(my_dict, key=my_dict.get))
    return new_classifications


if __name__ == "__main__":
    print get_majority([["r","y","r","r"],
                        ["y","r","r","r"],
                        ["y","y","y","y"],
                        ["y","y","r","r"]])
    import modulemanager
    modulemanager.ModuleManager()