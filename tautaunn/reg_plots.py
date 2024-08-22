# coding: utf-8

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# generic plot function for a single field in data
def field_plot(data, field, **kwargs):
    fig, ax = plt.subplots()
    for skim_name, arr in data.items():
        ax.hist(arr[field], bins=100, label=skim_name, histtype="step", density=True)
    return fig
