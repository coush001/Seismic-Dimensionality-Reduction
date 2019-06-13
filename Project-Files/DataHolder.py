import numpy as np
import copy

from sklearn.linear_model import LinearRegression

from utils import *

from DataHolder import *
from ModelAgent import *


class DataHolder:
    def __init__(self, field_name, inlines, xlines):

        # User input attributes
        self.field_name = field_name
        self.inlines = inlines
        self.xlines = xlines

        # near and far offset seismic data
        self.near = None
        self.far = None
        self.twt = None

        #  Dictionaries for multiple possible entries
        self.horizons = {}
        self.wells = {}

    def add_segy(self, name, fname):
        if name == 'near':
            self.near, twt = load_seismic(fname, inlines=self.inlines, xlines=self.xlines)
        elif name == 'far':
            self.far, twt = load_seismic(fname, inlines=self.inlines, xlines=self.xlines)
        else:
            raise Exception('please specify if near or far data')

        if self.twt is None:
            self.twt = twt
        else:
            assert (self.twt == twt).all, "This twt does not match the twt from the previous segy"

    def add_horizon(self, horizon_name, fname):
        self.horizons[horizon_name] = interpolate_horizon(load_horizon(fname, inlines=self.inlines, xlines=self.xlines))

    def add_well(self, well_id, well_i, well_x):
        self.wells[well_id] = [well_i, well_x]


class Processor:
    def __init__(self, near=None, far=None, twt=None):
        self.raw = [near, far]
        self.twt = twt
        self.out = None

    def flatten(self, data, horizon, top_add=12, below_add=52):
        out = []
        for amplitude in data:
            traces = np.zeros((horizon.shape[0], horizon.shape[1], top_add + below_add))
            for i in range(horizon.shape[0]):
                hrz_idx = [np.abs(self.twt - val).argmin() for val in horizon[i, :]]
                for j in range(horizon.shape[1]):
                    traces[i, j, :] = amplitude[hrz_idx[j] - top_add:hrz_idx[j] + below_add, i, j]
            out.append(traces)
        return out

    def normalise(self, data, well_i=38, well_x=138):
        out = []
        for i in data:
            well_variance = np.mean(np.std(i[well_i - 2:well_i + 1, well_x - 2:well_x + 1], 2))
            i /= well_variance
            out.append(i)

        return out

    def to_2d(self, data):
        return [i.reshape(-1, data[0].shape[-1]) for i in data]

    def average_neighbours(self, neighbours=10):
        return 'not implemented yet'

    def stack_traces(self, data):
        return np.concatenate([i for i in data], 1)

    @property
    def FF(self):
        x_avo = self.out[0]
        y_avo = self.out[1] - self.out[0]

        lin_reg = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        lin_reg.fit(x_avo.reshape(-1, 1), y_avo.reshape(-1, 1))

        print("Linear Regression coefficient: %1.2f" % lin_reg.coef_[0, 0])
        return y_avo - lin_reg.coef_ * x_avo

    def __call__(self, flatten=False, normalise=False, label='FF'):
        self.out = copy.copy(self.raw)

        if flatten:
            self.out = self.flatten(self.out, flatten[0], flatten[1], flatten[2])
        if normalise:
            self.out = self.normalise(self.out, normalise[0], normalise[1])

        #  Flatten arrays from 3d to 2d
        self.out = self.to_2d(self.out)

        # Pre-stacking calculation of FF
        if label == 'FF':
            self.label = self.FF

        #  Stack the traces for output
        self.out = self.stack_traces(self.out)

        return self.out, self.label