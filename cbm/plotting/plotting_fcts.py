import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_multiple_runs(results, x_name, x_values, y_name, predictors, save=False, savepath=None):
    # Reformat data into dataframe for plotting
    n_seeds, n_x_values = results.shape

    model_data = np.repeat(predictors, n_seeds * n_x_values)
    x_values_data = np.tile(np.repeat([str(x) for x in x_values], n_seeds), len(predictors))

    y_data = np.empty(shape=model_data.shape)
    # Loop over seeds
    for j in range(n_x_values):
        # Loop over x values
        for i in range(n_seeds):
            for k, predictor in enumerate(predictors):
                # Just a fancy way of flattening an array of dicts of dicts
                y_data[i+j*n_x_values+k*n_seeds*n_x_values] = results[i, j][predictor][y_name]

    plot_df = pd.DataFrame({'model': model_data,
                            x_name: x_values_data,
                            y_name: y_data})

    fig = plt.figure()
    ax = sns.lineplot(data=plot_df, x=x_name, y=y_name, hue='model')
    fig.show()
