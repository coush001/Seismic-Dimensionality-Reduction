import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from datashader.bokeh_ext import InteractiveImage
import bokeh.plotting as bp
import segypy
from scipy.interpolate import interp2d
from utils import load_seismic, load_horizon, colorbar, interpolate_horizon, plot_section_horizon_and_well
from utils import flatten_on_horizon

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

        self.near_traces_emb = self.near_traces.reshape(-1, 64)
        self.far_traces_emb = self.far_traces.reshape(-1, 64)


test = MasterClass('TEST_OBJECT')

# Loading Data
test.load_near('../data/3d_nearstack.sgy')
test.load_far('../data/3d_farstack.sgy')
test.load_horizon('../data/Top_Heimdal_subset.txt')

test.plot_horizon()

# add well location
test.add_well('well_1', 36, 276//2)

# print well dictionary and plot
print("Dictionary of wells", test.wells)
test.plot_horizon_well('well_1')

# flatten
test.flatten()

#

