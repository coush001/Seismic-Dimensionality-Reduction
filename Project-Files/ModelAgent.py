import numpy as np
import matplotlib.pyplot as plt

# Dim reduction tools
import umap
import torch
import torch.utils.data
from torch import optim

from torch.utils.data import TensorDataset
from sklearn.model_selection import ShuffleSplit

from utils import *


class ModelAgent:
    def __init__(self, data):
        self.input = data[0]
        self.label = data[1]
        print("ModelAgent initialised")

    def plot_2d(self, data, label):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sc = ax.scatter(data[:, 0], data[:, 1], s=2.0, c=np.min(label, 1))
        #         colorbar(sc)
        return 1

    def plot_3d(self, data, feature):
        return 'Not implemented'


class UMAP(ModelAgent):
    def __init__(self, data):
        super().__init__(data)

    def reduce(self, n_neighbors=50, min_dist=0.001):
        embedding_stack_ff = umap.UMAP(n_neighbors=n_neighbors,
                                       min_dist=min_dist,
                                       metric='correlation',
                                       verbose=False,
 #   this was in ASAP notebook: random_state=42).fit_transform(np.concatenate([stacked, FF.reshape(-1, 64)], 1))
                                       random_state=42).fit_transform(self.input)

        return embedding_stack_ff


class VAE_model(ModelAgent):
    def __init__(self, data):
        super().__init__(data)

    def create_dataloader(self, batch_size=32):
        #  split the concatenated input back into two arrays
        X = torch.from_numpy(np.stack(np.split(self.input, 2, axis=1), 1)).float()
        # Create a stacked representation and a zero tensor so we can use the standard Pytorch TensorDataset
        y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()

        print('adf', X.shape)

        split = ShuffleSplit(n_splits=1, test_size=0.5)
        for train_index, test_index in split.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

        train_dset = TensorDataset(X_train, y_train)
        test_dset = TensorDataset(X_test, y_test)
        all_dset = TensorDataset(X, y)

        kwargs = {'num_workers': 1, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, **kwargs)
        self.all_loader = torch.utils.data.DataLoader(all_dset, batch_size=batch_size, shuffle=False, **kwargs)

    def train_vae(self, cuda=False, epochs=30):
        set_seed(42)  # Set the random seed
        self.model = VAE(hidden_size=8)  # Inititalize the model

        # use cuda if chosen
        if cuda:
            self.model.cuda()

        # Create a gradient descent optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2, betas=(0.9, 0.999))

        # Store and plot losses
        self.losses = []

        # Start training loop
        for epoch in range(1, epochs + 1):
            tl = train(epoch, self.model, optimizer, self.train_loader, cuda=False)  # Train model on train dataset
            testl = test(epoch, self.model, self.test_loader, cuda=False)  # Validate model on test dataset
            self.losses.append([tl, testl])

    def run_vae(self):
        _, self.zs = forward_all(self.model, self.all_loader, cuda=False)

    def vae_umap(self):
        transformer = umap.UMAP(n_neighbors=5,
                                min_dist=0.001,
                                metric='correlation', verbose=True).fit(self.zs.numpy())
        embedding = transformer.transform(self.zs.numpy())
        print('shape of zs', self.zs.shape)

        # plot umap
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sc = ax.scatter(embedding[::, 0], embedding[::, 1], s=2.0, c=np.min(self.label, 1)[::])
        plt.show()

    def reduce(self):
        self.create_dataloader()
        self.train_vae()
        self.run_vae()
        self.vae_umap()


