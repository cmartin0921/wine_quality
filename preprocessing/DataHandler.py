import numpy as np
import pandas as pd

import sklearn.preprocessing as preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold


class DataHandler():

    def __init__(self, random_state=None):
        # self._df = None
        self._X = None
        self._y = None

        self.X_train = None
        self.y_train = None
        self.X_cross_validation = None
        self.y_cross_validation = None
        self.X_test = None
        self.y_test = None

        self._seed = random_state

    def load_data(self, data, features=None, target=None):
        self._X = data[features]
        self._y = data[target]

    def split_dataset(self, cross_validation_size=0.2, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=self._seed)

        if cross_validation_size != 0.0:
            self.X_train, self.X_cross_validation, self.y_train, self.y_cross_validation = train_test_split(
                self.X_train, self.y_train, test_size=test_size, random_state=self._seed)

    def apply_standardization(self):
        sc = preprocess.StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.fit_transform(self.X_test)

        if self.X_cross_validation is not None:
            self.X_cross_validation = sc.fit_transform(self.X_cross_validation)

    # def apply_kfold(self, num_folds=5):
    #     return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self._seed)

    def __str__(self):
        train_unique = np.unique(self.y_train, return_counts=True)

        return ("X_train shape: \t\t\t{a.X_train.shape}\t |\ty_train.shape: {a.y_train.shape}\nX_cross_validation shape: \t{a.X_cross_validation.shape}\t |\ty_cross_validation shape: {a.y_cross_validation.shape}\nX_test shape: \t\t\t{a.X_test.shape}\t |\ty_test shape: {a.y_test.shape}\n".format(a=self)
                )
