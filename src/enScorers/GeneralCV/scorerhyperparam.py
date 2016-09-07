# The scorerhyperparam.py file must contain the dictionary "hyperparam" and present the various hyperparameters in this
# scorer. If type is string, must include the entry "choices" with a list of possible strings.
# The modelhyperparam.py file must contain the dictionary "hyperparam" and present the various hyperparameters in this
# model. If type is string, must include the entry "choices" with a list of possible strings.
hyperparam = {"N": {"description": "Used to designate number of folds used for N-fold CV.",
                    "type": "int",
                    "default": 3
                    },

              "folds": {"description": "Use to determine method of partitioning the data. Use stratified for "
                                       "Stratified N-fold CV. Use regular for N-fold CV.",
                        "type": "string",
                        "choices": ["stratified", "regular"],
                        "default": "stratified"
                        },

              "shuffle": {"description": "If the data ordering is not arbitrary (e.g. samples with the same label are "
                                         "contiguous), shuffling it first may be essential to get a meaningful cross "
                                         "validation result. However, the opposite may be true if the samples are not "
                                         "independently and identically distributed. For example, if samples "
                                         "correspond to news articles, and are ordered by their time of publication, "
                                         "then shuffling the data will likely lead to a model that is overfit and an "
                                         "inflated validation score: it will be tested on samples that are "
                                         "artificially similar (close in time) to training samples.",
                          "type": "bool",
                          "default": False
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
