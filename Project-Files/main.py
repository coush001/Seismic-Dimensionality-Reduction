import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import bokeh.plotting as bp
import segypy
from scipy.interpolate import interp2d
from utils import load_seismic, load_horizon, colorbar, interpolate_horizon, plot_section_horizon_and_well
from utils import flatten_on_horizon

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from sklearn.model_selection import ShuffleSplit
from utils import VAE, set_seed
import os
from livelossplot import PlotLosses

print("works")


class MasterClass:

    def __init__(self, name):
        self.name = name

        self.near_stack_amplitudes = []
        self.far_stack_amplitudes = []
        self.twt_n = []
        self.twt_f = []

        self.horizon = []
        self.interpolated_horizon = []

        self.wells = {}

        self.near_traces = []
        self.far_traces = []

        self.near_traces_emb = []
        self.far_traces_emb = []

        self.stacked = np.empty(0)

        self.FF = np.empty(0)

        self.embedding_stack_ff = []

    def load_near(self, near_sgy):
        self.near_stack_amplitudes, self.twt_n = load_seismic(near_sgy, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2])

    def load_far(self, far_sgy):
        self.far_stack_amplitudes, self.twt_f = load_seismic(far_sgy, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2])

    def load_horizon(self, horizon_txt):
        self.horizon = load_horizon(horizon_txt, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2])

    def interpolate_horizon(self):
        self.interpolated_horizon = interpolate_horizon(self.horizon)

    def plot_horizon(self):
        self.interpolate_horizon()
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 12))
        ax1 = ax.imshow(self.interpolated_horizon, vmin=2030, vmax=2200)
        colorbar(ax1)
        plt.show()

    def add_well(self, well_id, well_x, well_i):
        self.wells[well_id] = [well_x, well_i]

    def plot_horizon_well(self, well_id):
        """
        Input: well_id
        Function : Plots given horizon and well 2D slice
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        plot_section_horizon_and_well(ax, self.near_stack_amplitudes, self.horizon, self.twt_n,
                                      inline=self.wells[well_id][0], well_pos=self.wells[well_id][1])
        plt.show()

    def flatten_traces(self):
        self.near_traces = flatten_on_horizon(self.near_stack_amplitudes, self.horizon, self.twt_n,
                                              top_add=12, below_add=52)
        self.far_traces = flatten_on_horizon(self.far_stack_amplitudes, self.horizon, self.twt_n,
                                             top_add=12, below_add=52)
        # TODO should these be here?
        np.save("near_traces_64.npy", self.near_traces)
        np.save("far_traces_64.npy", self.far_traces)

    def normalise_from_well(self, well_i=38, well_x=138):
        # TODO needed? maybe a save load func?
        self.near_traces = np.load("./near_traces_64.npy")
        self.far_traces = np.load("./far_traces_64.npy")

        # TODO sort this well functionality
        #well_i, well_x = 38, 138

        well_variance_near = np.mean(np.std(self.near_traces[well_i - 2:well_i + 1, well_x - 2:well_x + 1], 2))
        well_variance_far = np.mean(np.std(self.far_traces[well_i - 2:well_i + 1, well_x - 2:well_x + 1], 2))

        self.near_traces /= well_variance_near
        self.far_traces /= well_variance_far

        self.near_traces_emb = self.near_traces.reshape(-1, 64)  # 2d arrays of traces
        self.far_traces_emb = self.far_traces.reshape(-1, 64)

    def stack_traces(self, near=None, far=None):
        if not near:
            near = self.near_traces_emb
        if not far:
            far = self.far_traces_emb
        self.stacked = np.concatenate([near, far], 1)

    def get_FF(self, plot=True):
        x_avo = self.near_traces_emb
        y_avo = self.far_traces_emb - self.near_traces_emb

        lin_reg = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        lin_reg.fit(x_avo.reshape(-1, 1), y_avo.reshape(-1, 1))

        print("Linear Regression coefficient: %1.2f" % lin_reg.coef_[0, 0])
        self.FF = y_avo - lin_reg.coef_ * x_avo

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            sc = ax.imshow(np.min(self.FF.reshape(self.near_traces.shape[0],
                                                  self.near_traces.shape[1], 64)[:, :, :], 2))
            colorbar(sc)

    def umap(self):
        print(type(self.stacked), type(self.FF), self.stacked.shape, self.FF.reshape(-1, 64).shape)
        self.embedding_stack_ff = umap.UMAP(n_neighbors=50,
                                       min_dist=0.001,
                                       metric='correlation',
                                       verbose=True,
                                       random_state=42).fit_transform(np.concatenate([self.stacked,
                                                                                      self.FF.reshape(-1, 64)], 1))

    def plot_cluster(self, overlay=None):
        if not overlay:
            overlay = self.FF
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sc = ax.scatter(self.embedding_stack_ff[::, 0], self.embedding_stack_ff[::, 1], s=2.0, c=np.min(overlay, 1)[::])
        colorbar(sc)
        plt.show()



test = MasterClass('TEST_OBJECT')

# # Loading Data
# test.load_near('../data/3d_nearstack.sgy')
# test.load_far('../data/3d_farstack.sgy')
# test.load_horizon('../data/Top_Heimdal_subset.txt')
#
# test.plot_horizon()
#
# # add well location
# test.add_well('well_1', 36, 276//2)
#
# # print well dictionary and plot
# print("Dictionary of wells", test.wells)
# test.plot_horizon_well('well_1')
#
# # flatten
# test.flatten_traces()

# normalise and reshape to 2d arrays
test.normalise_from_well()

# linear regression to find FF:
test.get_FF()

# Stack the near and far on top of eachother
test.stack_traces()

# Umap dimensionality reduction:
test.umap()
test.plot_cluster()

