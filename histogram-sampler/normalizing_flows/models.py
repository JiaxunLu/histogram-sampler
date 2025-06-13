import torch
import numpy as np
import normflows as nf
from transforms import uniform
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import tqdm

class real_nvp:
    """
    Real NVP (Real-valued Non-Volume Preserving) normalizing flow.

    Strengths:
    - Tractable log-likelihood: can be trained with exact maximum likelihood.
    - Fast sampling and inversion due to simple coupling layers.
    - Scales well to higher dimensions.

    Weaknesses:
    - Affine coupling layers are limited in expressiveness.
    - Not suitable for disconnected, sharp, or highly non-linear/multimodal distributions.
    """
    def __init__(self, hist: np.histogramdd) -> None:
        # Wrap histogram data using uniform sampling abstraction
        u = uniform(hist)
        self.dim = u.dimension  # Dimension of the data
        self.base = nf.distributions.base.DiagGaussian(self.dim)  # Base distribution: standard normal
        self.target = torch.tensor(u.get_data(), dtype=torch.float32)  # Target distribution samples
        self.model = None
        self.loss_hist = None
        print(self.target.shape)
        
        dataset = TensorDataset(self.target)
        self.data_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

    def train(self, num_layers=32, max_iter=4000):
        # Real NVP flow built from Affine Coupling Blocks and swaps (permutations)
        flows = []
        for i in range(num_layers):
            # Neural net that predicts affine transform parameters for coupling layers
            # Only uses half the dimensions as input
            param_map = nf.nets.MLP([self.dim // 2, 64, 64, self.dim], init_zeros=True)
            
            # Add an affine coupling transformation
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            
            # Swap dimensions to alternate coupling inputs and outputs
            flows.append(nf.flows.Permute(self.dim, mode='swap'))

        # Compose the full normalizing flow model
        model = nf.NormalizingFlow(self.base, flows)

        loss_hist = np.array([])
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

        for it in range(max_iter):
            print(it)
            for batch in self.data_loader:
                optimizer.zero_grad()

                x = batch[0].detach().clone()
                
                # Kullback-Leibler divergence loss (forward KL from data to model)
                loss = model.forward_kld(x)

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()

                # loss_hist = np.append(loss_hist, loss.data.numpy())

        self.loss_hist = loss_hist
        self.model = model

    def sample(self, num=1000):
        # Sample from trained flow model
        return self.model.sample(num)



class AutoregressiveRationalQuadraticSpline:
    """
    Autoregressive Rational Quadratic Spline Flow.

    Strengths:
    - Very expressive: spline transformations can model sharp or curved density features.
    - Well-suited for disconnected or multimodal distributions.
    - Handles complex nonlinear dependencies through autoregressive transforms.

    Weaknesses:
    - Sampling is slower than Real NVP due to autoregressive structure.
    - Computationally heavier to train.
    """
    def __init__(self, hist: np.histogramdd) -> None:
        # Wrap histogram data using uniform sampling abstraction
        u = uniform(hist)
        self.dim = u.dimension
        self.base = nf.distributions.base.DiagGaussian(self.dim)  # Base = standard normal
        target = torch.tensor(u.get_data(), dtype=torch.float32)
        self.model = None
        self.loss_hist = None

        self.mu = target.mean()
        self.std = target.std()
        self.target_normalized = (target - self.mu) / self.std

        self.dataset = TensorDataset(self.target_normalized)

    def train(self, 
              max_iter=4000, 
              lr=5e-4, 
              weight_decay=5e-5, 
              hidden_units=128, 
              hidden_layers=2, 
              K=16, 
              batch_size=1000, 
              shuffle=True,
              show_progress=True,
              ):
        # Build flow using neural spline transforms
        # K: Number of flow steps
        # hidden_units: Width of hidden layers
        # hidden_layers: Number of hidden layers per transform

        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        flows = []
        for i in range(K):
            # Autoregressive Rational Quadratic Spline flow
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.dim, hidden_layers, hidden_units)]
            # Learnable permutation to increase expressivity
            if self.dim > 1:
                flows += [nf.flows.LULinearPermute(self.dim)]

        model = nf.NormalizingFlow(self.base, flows)

        loss_hist = []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for it in tqdm.trange(max_iter, desc='Training') if show_progress else range(max_iter):
            total_loss = 0.0
            for batch in data_loader:
                optimizer.zero_grad()
                x = batch[0].clone().detach()
                loss = model.forward_kld(x)  # KL divergence loss

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.data.numpy()

            loss_hist.append(total_loss)

        self.loss_hist = np.array(loss_hist)
        self.model = model

    def sample(self, num=1000):
        return self.model.sample(num)[0].detach() * self.std + self.mu


