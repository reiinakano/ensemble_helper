# All scorer.py files must contain the function score(), which takes as input the following items: an instance of one of
# the models of a modelclass.py file, a feature set, the corresponding correct labels of the feature set, and the set of
# "hyperparameters" of the scoring function. It then returns a float value indicating the score of the model using the
# feature set.

# NOTES ABOUT THIS PARTICULAR SCORER
# This scorer uses N-fold cross validation and returns the average ACCURACY of all N folds
def score(model, feature_set, labels, N):
