# This module contains the FeatureSet class, used to store relevant parameters about a certain feature set, such as its
# parent set, the column numbers of the features included in the feature set, and an optional user-supplied definition
# of the feature set.
# As of now, the FeatureSet class is mainly used to choose particular features from a ParentSet e.g. Naive Bayes will
# work better with a feature set with less correlated features.
# I plan to rewrite both FeatureSet and ParentSet using pandas dataframes when ensemble-helper starts incorporating
# feature engineering.


class FeatureSet:
    # parent_set is the ParentSet instance from which this particular FeatureSet was derived.
    # As of now, ParentSet is simply an object containing the attributes "features" and "labels" which are numpy arrays
    # that correspond to the, uh, features and labels.
    # feature_indices is a list containing the column numbers of the particular features that are included in this
    # feature set.
    # features is a 2D numpy array containing the features of the feature set. This is obtained from parent_set and
    # feature_indices.
    # labels contains the labels of the feature set. Since feature sets (as of now) don't remove rows from the parent
    # set, labels is equal to the labels of the parent set.
    def __init__(self, parent_set, feature_indices):
        self.parent_set = parent_set
        self.feature_indices = feature_indices
        self.description = ""  # User supplied description e.g. "This feature set works best with logistic regression!"
        self.features = parent_set.features[:, feature_indices]
        self.labels = parent_set.labels

    # This method returns the features of the feature set in scikit-learn friendly form (numpy array)
    def get_features_array(self):
        return self.features

    # This method returns the features of the feature set in scikit-learn friendly form (numpy array)
    def get_labels_array(self):
        return self.labels


if __name__ == "__main__":
    pass
