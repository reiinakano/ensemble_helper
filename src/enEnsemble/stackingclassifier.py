# This class contains the class for implementing a stacking classifier from base models (NOT model versions)
from sklearn.cross_validation import StratifiedKFold
import numpy as np


class StackingClassifier:
    def __init__(self, basemodels, secondary_model, n_folds=2, use_features_in_secondary=False, use_predict_proba=False):
        self.basemodels = basemodels
        self.secondary_model = secondary_model
        self.n_folds = n_folds
        # This boolean decides whether to include the original features (not the base model predictions) in fitting and
        # predicting with the secondary model
        self.use_features_in_secondary = use_features_in_secondary
        self.use_predict_proba = use_predict_proba

    def fit(self, features, labels):
        skf = StratifiedKFold(labels, n_folds=self.n_folds)

        all_model_predictions = np.array([]).reshape(len(labels), 0)
        for model in self.basemodels:
            if not self.use_predict_proba:
                single_model_prediction = np.array([]).reshape(0, 1)
            else:
                single_model_prediction = np.array([]).reshape(0, len(set(labels)))

            for train_index, test_index in skf:
                if not self.use_predict_proba:
                    prediction = model.fit(features[train_index], labels[train_index]).predict(features[test_index])
                    prediction = prediction.reshape(prediction.shape[0], 1)
                else:
                    prediction = model.fit(features[train_index], labels[train_index]).predict_proba(features[test_index])
                single_model_prediction = np.vstack([single_model_prediction.astype(prediction.dtype), prediction])

            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))

        # We have to shuffle the labels in the same order as we generated predictions during CV
        # (we kinda shuffled them when we did Stratified CV)
        # We also do the same with the features (we will need this only IF use_features_in_secondary is True)
        reordered_labels = np.array([]).astype(labels.dtype)
        reordered_features = np.array([]).reshape((0, features.shape[1])).astype(features.dtype)
        for train_index, test_index in skf:
            reordered_labels = np.concatenate((reordered_labels, labels[test_index]))
            reordered_features = np.concatenate((reordered_features, features[test_index]))

        #print reordered_features
        #print all_model_predictions
        #print reordered_labels
        #print np.vstack((all_model_predictions.T, reordered_labels)).T[:100]
        #print reordered_features
        #print np.hstack((reordered_features, all_model_predictions))

        # Fit the base models correctly this time using ALL the training set
        for model in self.basemodels:
            model.fit(features, labels)

        # Fit the secondary model
        if not self.use_features_in_secondary:
            self.secondary_model.fit(all_model_predictions, reordered_labels)
        else:
            self.secondary_model.fit(np.hstack((reordered_features, all_model_predictions)), reordered_labels)

        return self

    def predict(self, features_to_predict):
        all_model_predictions = np.array([]).reshape(len(features_to_predict), 0)
        for model in self.basemodels:
            if not self.use_predict_proba:
                single_model_prediction = model.predict(features_to_predict)
                single_model_prediction = single_model_prediction.reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(features_to_predict)
            all_model_predictions = np.hstack((all_model_predictions.astype(single_model_prediction.dtype), single_model_prediction))
        if not self.use_features_in_secondary:
            return self.secondary_model.predict(all_model_predictions)
        else:
            return self.secondary_model.predict(np.hstack((features_to_predict, all_model_predictions)))




if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import datasets
    iris = datasets.load_digits()
    X = iris.data
    y = iris.target
    clf = StackingClassifier([LogisticRegression(), RandomForestClassifier(n_estimators=30, random_state=32), KNeighborsClassifier()], LogisticRegression(), 2, False, True)
    clf.fit(X, y)
    #clf.predict(X)

    from sklearn.cross_validation import cross_val_score
    print np.mean(cross_val_score(LogisticRegression(), X, y))
    print np.mean(cross_val_score(RandomForestClassifier(n_estimators=30), X, y))
    print np.mean(cross_val_score(KNeighborsClassifier(), X, y))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    labels = y
    model = clf
    feature_set = X
    skf = StratifiedKFold(labels, n_folds=10)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    confusion_matrices = []
    for train, test in skf:
        X_train, X_test, y_train, y_test = feature_set[train], feature_set[test], labels[train], labels[test]
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))
        precision.append(precision_score(y_test, prediction, pos_label=None, average='weighted'))
        recall.append(recall_score(y_test, prediction, pos_label=None, average='weighted'))
        f1.append(f1_score(y_test, prediction, pos_label=None, average='weighted'))
        confusion_matrices.append(confusion_matrix(y_test, prediction))
    metrics = {}
    metrics["accuracy"] = np.mean(accuracy)
    metrics["precision"] = np.mean(precision)
    metrics["recall"] = np.mean(recall)
    metrics["f1"] = np.mean(f1)
    metrics["confusion_matrix"] = confusion_matrices[0]
    for matrix in confusion_matrices[1:]:
        metrics["confusion_matrix"] += matrix
    print metrics
    print accuracy
