# This module contains the FeatureSet class, used to store relevant parameters about a certain feature set, such as its
# parent set, the column numbers of the features included in the feature set, and an optional user-supplied definition
# of the feature set.


class FeatureSet:
    # parent_set is the ParentSet instance from which this particular FeatureSet was derived.
    # As of now, ParentSet is simply an object containing the attributes "features" and "labels" which are numpy arrays
    # that correspond to the, uh, features and labels.
    # feature_indices is a list containing the column numbers of the
    def __init__(self, parent_set, feature_indices):