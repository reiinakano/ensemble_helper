# This module contains the FeatureExtractor class, which must contain information about how to extract the feature set
# from any parent set.
# As of now, the FeatureExtractor class is mainly used to choose particular features from a ParentSet e.g. Naive Bayes will
# work better with a feature set with less correlated features.
# I plan to rewrite both FeatureExtractor and ParentSet using pandas dataframes when ensemble-helper starts incorporating
# feature engineering.


class FeatureExtractor:
    # parent_set is the ParentSet instance from which this particular FeatureSet was derived.
    # As of now, ParentSet is simply an object containing the attributes "features" and "labels" which are numpy arrays
    # that correspond to the, uh, features and labels.
    # feature_indices is a list containing the column numbers of the particular features that are included in this
    # feature set.
    # features is a 2D numpy array containing the features of the feature set. This is obtained from parent_set and
    # feature_indices.
    # labels contains the labels of the feature set. Since feature sets (as of now) don't remove rows from the parent
    # set, labels is equal to the labels of the parent set.
    def __init__(self, feature_indices):
        self.feature_indices = feature_indices
        self.description = ""  # User supplied description e.g. "This feature set works best with logistic regression!"

    # This method returns the features of parent_set in scikit-learn friendly form (numpy array)
    def get_features_array(self, parent_set):
        return parent_set.features[:, self.feature_indices]

    # This method returns the labels of parent_set in scikit-learn friendly form (numpy array)
    def get_labels_array(self, parent_set):
        return parent_set.labels


if __name__ == "__main__":
    import parentset
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    parent_set = parentset.ParentSet(X, y)
    a = FeatureExtractor(range(parent_set.features.shape[1]))
    print a.get_features_array(parent_set)
    print a.get_labels_array(parent_set)
