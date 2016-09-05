# This file contains the interface for interacting with the models and scorers in the packages enModels and enScorers
import pkgutil


class ModuleManager:
    # To initialize ModuleManager, it must crawl through the directories enModels and enScorers to determine which
    # models and scoring methods are correctly configured and available for use.
    def __init__(self):
        self.available_models = self._get_available_models()
        self.available_scorers = self._get_available_scorers()

    def _get_available_models(self):
        available_models = {}

        return available_models

    def _get_available_scorers(self):
        available_scorers = {}

        return available_scorers


if __name__ == "__main__":