# The modelhyperparam.py file must contain the dictionary "hyperparam" and present the various hyperparameters in this
# model. If type is string, must include the entry "choices" with a list of possible strings.
hyperparam = {"penalty": {"description": "Used to specify the norm used in the penalization. The newton-cg and lbfgs "
                                         "solvers support only l2 penalties.",
                          "type": "str",
                          "choices": ["l1", "l2"],
                          "default": "l2"
                          },

              "dual": {"description": "Dual or primal formulation. Dual formulation is only implemented for l2 penalty "
                                      "with liblinear solver. Prefer dual=False when n_samples > n_features.",
                       "type": "bool",
                       "default": False
                        },

              "C": {"description": "Inverse of regularization strength; must be a positive float. Like in support "
                                   "vector machines, smaller values specify stronger regularization.",
                    "type": "float",
                    "default": 1.0
                    },

              "fit_intercept": {"description": "Specifies if a constant (a.k.a. bias or intercept) should be added to "
                                               "the decision function.",
                                "type": "bool",
                                "default": True
                                },

              "intercept_scaling": {"description": "Useful only if solver is liblinear. When self.fit_intercept is "
                                                   "True, instance vector x becomes [x, self.intercept_scaling], i.e. "
                                                   "a synthetic feature with constant value equals to intercept_scaling"
                                                   " is appended to the instance vector.",
                                    "type": "float",
                                    "default": 1.0},

              "class_weight": {"description": "Weights associated with classes in the form {class_label: weight}. If "
                                              "not given, all classes are supposed to have weight one. The balanced "
                                              "mode uses the values of y to automatically adjust weights inversely "
                                              "proportional to class frequencies in the input data as n_samples / "
                                              "(n_classes * np.bincount(y))",
                               "type": "str",
                               "choices": [None, "balanced"],
                               "default": None},

              "max_iter": {"description": "Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of "
                                          "iterations taken for the solvers to converge.",
                           "type": "int",
                           "default": 100},

              "solver": {"description": "Algorithm to use in the optimization problem. For small datasets, liblinear "
                                        "is a good choice, whereas sag is faster for large ones. For multiclass "
                                        "problems, only newton-cg and lbfgs handle multinomial loss; sag and liblinear "
                                        "are limited to one-versus-rest schemes. newton-cg, lbfgs and sag only handle "
                                        "L2 penalty. Note that sag fast convergence is only guaranteed on features "
                                        "with approximately the same scale. You can preprocess the data with a scaler "
                                        "from sklearn.preprocessing",
                         "type": "str",
                         "choices": ["newton-cg", "lbfgs", "liblinear", "sag"],
                         "default": "liblinear"},

              "tol": {"description": "Tolerance for stopping criteria.",
                      "type": "float",
                      "default": 0.0001},

              "multi_class": {"description": "Multiclass option can be either ovr or multinomial. If the option chosen "
                                             "is ovr, then a binary problem is fit for each label. Else the loss "
                                             "minimised is the multinomial loss fit across the entire probability "
                                             "distribution. Works only for the lbfgs solver.",
                              "type": "str",
                              "choices": ["ovr", "multinomial"],
                              "default": "ovr"},

              "n_jobs": {"description": "Number of CPU cores used during the cross-validation loop. If given a value "
                                        "of -1, all cores are used.",
                         "type": "int",
                         "default": 1}
              }
