# This module contains the class for implementing a hard voting (uses majority classification, not average voting)
# ensemble.


class HardVotingClassifier:
    def __init__(self, parent_set, module_mgr):
        self.parent_set = parent_set
        self.module_mgr = module_mgr
        self.scores = {}  # scores dictionary
        self._modelversions = set([])  # contains the ModelVersions that are in the HardVotingClassifier
        self.user_notes = ""
        self.scored = False

    def add_model(self, modelversion):
        # Make sure only model versions with same parent set are in the ensemble
        if modelversion.parent_set is self.parent_set:
            self._modelversions.add(modelversion)

    def score(self, scorer_name, scorer_hyperparam):
        pass

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
