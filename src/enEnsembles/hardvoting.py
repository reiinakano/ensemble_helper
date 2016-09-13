# This module contains the class for implementing a hard voting (uses majority classification, not averaging) ensemble.


class HardVotingClassifier:
    def __init__(self, parent_set, scorer_name, scorer_hyperparam, module_mgr):
        self.parent_set = parent_set
        self.scorer_name = scorer_name
        self.scorer_hyperparam = scorer_hyperparam
        self.scorer_func = module_mgr.get_scorer_func(scorer_name)
        self.scores = {}
        self.models = {}