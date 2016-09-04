# Every modelclass.py file must contain its own ModelClass class.
# As of v0, ModelClass must implement the methods _train(), _predict(), _save_internals(), and static method _restore_model()
# Other methods (e.g. predict_proba()) may be added to provide better functionality for various scoring methods.
class ModelClass:
    # __init__() must take as parameters all hyperparameters required for properly implementing train() and predict()
    def __init__(self, penalty, dual, C, fit_intercept, intercept_scaling, class_weight, max_iter, solver, tol, multi_class, n_jobs):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(penalty=penalty, dual=dual, C=C, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        max_iter=max_iter, solver=solver, tol=tol, multi_class=multi_class,
                                        n_jobs=n_jobs)

    # _train() must take as parameters the feature set and correct labels to use as the training data.
    # If successful, it returns True. Else, it returns False
    def _train(self, feature_set, labels):
        try:
            self.model.fit(feature_set, labels)
        except:
            return False
        return True

    # _predict() must take as parameter the feature set to predict. Obviously, the feature set used must have the same
    #   format as the feature set used to train the model. Otherwise, behavior is undefined.
    # _predict() must return the predicted labels corresponding to the input.
    def _predict(self, feature_set_to_predict):
        return self.model.predict(feature_set_to_predict)

    # _save_internals() must take a filename as parameter to determine where to store a particular model
    # The method must be able to save all information required to fully restore a particular ModelClass instance when
    # calling the corresponding _restore_model() method.
    # If successful, it returns True. Else, it returns False
    def _save_internals(self, filename):
        try:
            from sklearn.externals import joblib
            joblib.dump(self, filename, compress=3)
        except:
            return False
        return True

    # This static method _restore_model() must take a filename and return the original ModelClass instance that was
    # saved there using the _save_internals() method.
    @staticmethod
    def _restore_model(filename):
        try:
            from sklearn.externals import joblib
            return joblib.load(filename)
        except:
            return None
