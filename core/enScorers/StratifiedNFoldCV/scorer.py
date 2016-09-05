# All scorer.py files must contain the function score(), which takes as input the following items: an instance of one of
# the models of a modelclass.py file, a feature set, the corresponding correct labels of the feature set, and the set of
# "hyperparameters" of the scoring function. It then returns a dictionary containing the model's scores for various
# performance metrics e.g. accuracy, precision, recall, etc.
# It must also include the function scorer_name(), returning the name of the scoring method used.
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


# NOTES ABOUT THIS PARTICULAR SCORER
# This scorer uses stratified N-fold cross validation and returns the averaged performance metrics of all N folds
def score(model, feature_set, labels, N=3, shuffle=False, calc_acc=True, calc_prc=True, calc_rec=True, calc_f1=True, calc_cm=True):
    assert len(labels) >= N
    skf = StratifiedKFold(labels, n_folds=N, shuffle=shuffle)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in skf:
        X_train, X_test, y_train, y_test = feature_set[train], feature_set[test], labels[train], labels[test]
        model.train(X_train, y_train)
        prediction = model.predict(X_test)
        if calc_acc:
            accuracy.append(accuracy_score(y_test, prediction))
        if calc_prc:
            precision.append(precision_score(y_test, prediction, pos_label=None, average='weighted'))
        if calc_rec:
            recall.append(recall_score(y_test, prediction, pos_label=None, average='weighted'))
        if calc_f1:
            f1.append(f1_score(y_test, prediction, pos_label=None, average='weighted'))
    metrics = {}
    if calc_acc:
        metrics["accuracy"] = np.mean(accuracy)
    if calc_prc:
        metrics["precision"] = np.mean(precision)
    if calc_rec:
        metrics["recall"] = np.mean(recall)
    if calc_f1:
        metrics["f1"] = np.mean(f1)
    if calc_cm:
        metrics["confusion_matrix"] = confusion_matrix(y_test, prediction)
    return metrics


def scorer_name():
    return "Stratified N-fold Cross Validation"
