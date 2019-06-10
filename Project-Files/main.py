import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils import load_seismic, load_horizon, colorbar, interpolate_horizon, plot_section_horizon_and_well
from utils import flatten_on_horizon


import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import ShuffleSplit
from utils import VAE, set_seed


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

        self.train_loader = 0
        self.test_loader = 0
        self.all_loader = 0

        self.model = 0
        self.recs = 0
        self.zs = 0

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

    def plot_umap(self, overlay=None):
        if not overlay:
            overlay = self.FF
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sc = ax.scatter(self.embedding_stack_ff[::, 0], self.embedding_stack_ff[::, 1], s=2.0, c=np.min(overlay, 1)[::])
        colorbar(sc)
        plt.show()

    def create_dataloader(self, batch_size=32):
        # Create a stacked representation and a zero tensor so we can use the standard Pytorch TensorDataset
        X = torch.from_numpy(np.stack([self.near_traces_emb, self.far_traces_emb], 1)).float()
        y = torch.from_numpy(np.zeros((X.shape[0], 1))).float()
        print(X.shape)

        # We do an 20, 80 split in this case, fairly large since there are so many traces to learn from
        # Should also try a different split of say 80, 20
        split = ShuffleSplit(n_splits=1, test_size=0.8)
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
        losses = []
        # liveloss = PlotLosses()
        min_loss = 999999.

        # Start training loop
        for epoch in range(1, epochs + 1):
            logs = {}
            tl = train(epoch, self.model, optimizer, self.train_loader, cuda=False)  # Train model on train dataset
            logs['' + 'log loss'] = tl

            testl = test(epoch, self.model, self.test_loader, cuda=False)  # Validate model on test dataset
            logs['val_' + 'log loss'] = testl

            losses.append([tl, testl])

            # # Update the lossplot
            # liveloss.update(logs)
            # liveloss.draw()

            # Store best validation loss model
            if testl < min_loss:
                torch.save(self.model.state_dict(), "./models/model_epoch_" + str(epoch) + ".pth")
                min_loss = testl

            # break

    def run_vae(self):
        b = VAE(hidden_size=8)
        b.load_state_dict(torch.load("./models/model_epoch_30.pth"))
        self.recs, self.zs = forward_all(b, self.all_loader, cuda=False)

    def vae_umap(self):
        transformer = umap.UMAP(n_neighbors=5,
                                min_dist=0.001,
                                metric='correlation', verbose=True).fit(self.zs.numpy())
        embedding = transformer.transform(self.zs.numpy())

        # plot umap
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sc = ax.scatter(embedding[::, 0], embedding[::, 1], s=2.0, c=np.min(self.FF, 1)[::])
        colorbar(sc)
        plt.show()


# TODO put the following into utils
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, window_size=64):
    criterion_mse = nn.MSELoss(size_average=False)
    MSE = criterion_mse(recon_x.view(-1, 2, window_size), x.view(-1, 2, window_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


# Function to perform one epoch of training
def train(epoch, model, optimizer, train_loader, cuda=False, log_interval=10):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)

        if cuda:
            data = data.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item() * data.size(0)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() * data.size(0) / len(train_loader.dataset)))

    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


# Function to perform evaluation of data on the model, used for testing
def test(epoch, model, test_loader, cuda=False, log_interval=10):
    model.eval()
    test_loss = 0
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item() * data.size(0)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# Function to forward_propagate a set of tensors and receive back latent variables and reconstructions
def forward_all(model, all_loader, cuda=False):
    model.eval()
    reconstructions, latents = [], []
    with torch.set_grad_enabled(False):
        for i, (data, _) in enumerate(all_loader):
            if cuda:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar, z = model(data)
            reconstructions.append(recon_batch.cpu())
            latents.append(z.cpu())
    return torch.cat(reconstructions, 0), torch.cat(latents, 0)

