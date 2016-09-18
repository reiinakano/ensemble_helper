# This class contains the class for implementing a majority voting classifier from base models (NOT model versions)
import numpy as np


class VotingClassifier:
    # basemodels is a list of base models composing VotingClassifier
    def __init__(self, basemodels, weights=None, use_predict_proba=False):
        self.basemodels = basemodels
        self.use_predict_proba = use_predict_proba
        self.classes_ = None
        if weights is None:
            self.weights = [1] * len(basemodels)
        else:
            self.weights = weights

    def fit(self, features, labels):
        for model in self.basemodels:
            model.fit(features, labels)
        self.classes_ = np.asarray(sorted(list(set(labels))))
        return self

    def predict(self, features_to_predict):
        classifications = []
        for model in self.basemodels:
            if not self.use_predict_proba:
                classifications.append(model.predict(features_to_predict))
            else:
                classifications.append(model.predict_proba(features_to_predict))
        classifications = VotingClassifier.get_majority(classifications, self.weights, self.classes_)
        return classifications

    @staticmethod
    def get_majority(classifications, weights, classes=None):
        new_classifications = []
        if classifications[0].ndim == 1:
            for votes in zip(*classifications):
                my_dict = {}
                for index, vote in enumerate(votes):
                    if vote not in my_dict:
                        my_dict[vote] = weights[index]
                    else:
                        my_dict[vote] += weights[index]
                new_classifications.append(max(my_dict, key=my_dict.get))
        else:
            for proba_classification, weight in zip(classifications, weights):
                new_classifications.append(proba_classification*weight)
            new_classifications = np.average(np.asarray(new_classifications), axis=0)
            new_classifications = [classes[np.argmax(arr)] for arr in new_classifications]

        return new_classifications