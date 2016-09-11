# This module contains the ParentSet class. As of now, it contains two attributes, features and labels, which contain
# the, uhh, features and labels.
import featureset


class ParentSet:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
