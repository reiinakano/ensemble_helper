# The scorerhyperparam.py file must contain the dictionary "hyperparam" and present the various hyperparameters in this
# scorer. If type is string, must include the entry "choices" with a list of possible strings.
# The modelhyperparam.py file must contain the dictionary "hyperparam" and present the various hyperparameters in this
# model. If type is string, must include the entry "choices" with a list of possible strings.
hyperparam = {"N": {"description": "Used to designate number of stratified folds used for stratified N-fold CV.",
                    "type": "int",
                    "default": 3
                    },

              "calc_acc": {"description": "Denotes if accuracy is calculated during each cross validation stage",
                           "type": "bool",
                           "default": True
                           },

              "calc_prc": {"description": "Denotes if precision is calculated during each cross validation stage",
                           "type": "bool",
                           "default": True
                           },

              "calc_rec": {"description": "Denotes if recall is calculated during each cross validation stage",
                           "type": "bool",
                           "default": True
                           },

              "calc_f1": {"description": "Denotes if f1 score is calculated during each cross validation stage",
                          "type": "bool",
                          "default": True
                          },

              "calc_cm": {"description": "Denotes if confusion matrix is calculated. Since confusion matrices can't be"
                                         " averaged, the confusion matrix will be calculated for the last stage of"
                                         " cross validation only",
                          "type": "bool",
                          "default": True
                          }
              }
