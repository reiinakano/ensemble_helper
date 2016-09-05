# This file contains the interface for interacting with the models and scorers in the packages enModels and enScorers
import os, importlib


class ModuleManager:
    # To initialize ModuleManager, it must crawl through the directories enModels and enScorers to determine which
    # models and scoring methods are correctly configured and available for use.
    def __init__(self):
        self.available_models = self._get_available_models()
        self.available_scorers = self._get_available_scorers()

    def _get_available_models(self):
        available_models = {}
        for dir_name in [f for f in os.listdir('./enModels') if os.path.isdir('./enModels/' + f)]:
            # insert instructions to test if path is a legitimate model package
            mod_name = 'enModels.' + dir_name + '.modelclass'
            imported_module = importlib.import_module(mod_name)
            my_class = getattr(imported_module, "ModelClass")
            model_name = my_class.model_name()


        return available_models

    def _get_available_scorers(self):
        available_scorers = {}

        return available_scorers


if __name__ == "__main__":
    print ['./enScorers/' + f for f in os.listdir('./enScorers') if os.path.isdir('./enScorers/' + f)]
    print ['./enModels/' + f for f in os.listdir('./enModels') if os.path.isdir('./enModels/' + f)]
    for dir_name in [f for f in os.listdir('./enModels') if os.path.isdir('./enModels/' + f)]:
        # insert instructions to test if path is a legitimate model package
        mod_name = 'enModels.' + dir_name + '.modelclass'
        print mod_name
        imported_module = importlib.import_module(mod_name)
        print imported_module
        my_class = getattr(imported_module, "ModelClass")
        print my_class
        print my_class.model_name()