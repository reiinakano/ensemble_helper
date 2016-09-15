class VotingClassifier:
    # basemodels is a list of base models composing VotingClassifier
    def __init__(self, basemodels, weights=None, voting="hard"):
        self.basemodels = basemodels
        if weights is None:
            self.weights = [1] * len(basemodels)
        else:
            self.weights = weights

    def fit(self, features, labels):
        for model in self.basemodels:
            model.fit(features, labels)

    def predict(self, features_to_predict):
        classifications = []
        for model in self.basemodels:
            classifications.append(model.predict(features_to_predict))
        classifications = VotingClassifier.get_majority(classifications, self.weights)
        return classifications

    @staticmethod
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