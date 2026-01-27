import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from cbm.estimation.base_regressor import BaseRegressor


class LinRegressor(BaseRegressor):
    def __init__(self, seed, d_micro_in, d_micro_out, d_bottleneck, source,
                 target, d_cond):
        super().__init__(seed, d_micro_in, d_micro_out, d_bottleneck, source,
                         target, d_cond)

        self.model = LinearRegression()

    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) == 0:  # no conditioning
            self.model.fit(X, Y)
        else:
            self.X_dim = X.shape[-1]  # need this later in get_bottleneck_fct

            X_cat = np.concatenate((X, X_cond), axis=1)
            self.model.fit(X_cat, Y)

            a = 0

    def get_bottleneck_and_mechanism_fcts(self):
        try:
            lin_map = self.model.coef_[:, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            lin_map = self.model.coef_.T
        bottleneck_lin_map = lin_map[:, :self.d_bottleneck]

        mechanism_lin_map = np.linalg.pinv(bottleneck_lin_map) @ lin_map

        def bottleneck_fct(x):
            return x @ bottleneck_lin_map

        def mechanism_fct(x):
            return x @ mechanism_lin_map

        return bottleneck_fct, mechanism_fct


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

    def get_bottleneck_and_mechanism_fcts(self):
        U = self.pca.components_
        red_components = U.T @ U @ self.model.coef_

        try:
            bottleneck_components = red_components[:, :self.X_dim].T
            lin_map = self.model.coef_[:, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            bottleneck_components = red_components.T
            lin_map = self.model.coef_.T
        bottleneck_lin_map = bottleneck_components[:, :self.d_bottleneck]

        mechanism_lin_map = np.linalg.pinv(bottleneck_lin_map) @ lin_map

        def bottleneck_fct(x):
            return x @ bottleneck_lin_map

        def mechanism_fct(x):
            return x @ mechanism_lin_map

        return bottleneck_fct, mechanism_fct
