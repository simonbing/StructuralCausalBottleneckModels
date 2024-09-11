import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from cbm.estimation.base_regressor import BaseRegressor


class LinRegressor(BaseRegressor):
    def __init__(self, seed, d_micro_in, d_micro_out, d_bottleneck, d_cond):
        super().__init__(seed, d_micro_in, d_micro_out, d_bottleneck, d_cond)

        self.model = LinearRegression()

    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) == 0:  # no conditioning
            self.model.fit(X, Y)
        else:
            self.X_dim = X.shape[-1]  # need this later in get_bottleneck_fct

            X_cat = np.concatenate((X, X_cond), axis=1)
            self.model.fit(X_cat, Y)

    def get_bottleneck_fct(self):
        # using the first d_bottleneck entries...double check if this makes sense
        # TODO: this should be the d_bottleneck lin indep rows!
        try:
            # print(self.model.coef_)
            linear_map = self.model.coef_[:self.d_bottleneck, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            linear_map = self.model.coef_[:self.d_bottleneck, :].T

        # print(f'Linear map:\n{linear_map}')
        fct = lambda x: x @ linear_map

        return fct


class ReducedRankRegressor(LinRegressor):
    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) == 0:  # no conditioning
            self.model.fit(X, Y)

            Y_hat = self.model.predict(X)
        else:
            self.X_dim = X.shape[-1]  # need this later in get_bottleneck_fct

            X_cat = np.concatenate((X, X_cond), axis=1)
            self.model.fit(X_cat, Y)

            Y_hat = self.model.predict(X_cat)

        self.pca = PCA(n_components=self.d_bottleneck)
        self.pca.fit(Y_hat)

    def get_bottleneck_fct(self):
        U = self.pca.components_
        red_components = U.T @ U @ self.model.coef_
        try:
            linear_map = red_components[:self.d_bottleneck, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            linear_map = red_components[:self.d_bottleneck, :].T
        # print(f'Linear map:\n{linear_map}')
        fct = lambda x: x @ linear_map

        return fct
