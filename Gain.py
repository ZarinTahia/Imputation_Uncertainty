import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy as np
import torch
from geomloss import SamplesLoss

from Utils import nanmean, MAE, RMSE

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

class GAINImputer:
    def __init__(self, hint_rate=0.9, alpha=100, batch_size=128, epochs=1000, lr=1e-3, device=None):
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    class Generator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim*2, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, input_dim)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x, m):
            inputs = torch.cat([x, m], dim=1)
            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            return self.sigmoid(self.fc3(x))
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim*2, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, input_dim)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x, h):
            inputs = torch.cat([x, h], dim=1)
            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            return self.sigmoid(self.fc3(x))
    
    def fit_transform(self, X):
            # Convert to tensor
            if not torch.is_tensor(X):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X = X.detach().clone().float().to(self.device)
            self.mask = (~torch.isnan(X)).float()
            self.missing_mask = torch.isnan(X)

            # Initialize missing values with random uniform
            self.X_init = X.clone()
            self.X_init[self.missing_mask] = torch.rand(self.missing_mask.sum()).to(self.device)

            n, d = X.shape
            self.G = self.Generator(d).to(self.device)
            D = self.Discriminator(d).to(self.device)

            G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr)
            D_optimizer = optim.Adam(D.parameters(), lr=self.lr)

            for epoch in range(self.epochs):
                idx = torch.randperm(n)
                for i in range(0, n, self.batch_size):
                    batch_idx = idx[i:i+self.batch_size]
                    X_mb = self.X_init[batch_idx]
                    M_mb = self.mask[batch_idx]

                    # Discriminator
                    Z_mb = torch.rand_like(X_mb)
                    X_tilde = M_mb * X_mb + (1 - M_mb) * Z_mb
                    H_mb = M_mb * torch.bernoulli(torch.full_like(M_mb, self.hint_rate))
                    with torch.no_grad():
                        G_sample = self.G(X_tilde, M_mb)
                    Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
                    D_prob = D(Hat_X, H_mb)
                    D_loss = -torch.mean(M_mb*torch.log(D_prob+1e-8) + (1-M_mb)*torch.log(1-D_prob+1e-8))
                    D_optimizer.zero_grad(); D_loss.backward(); D_optimizer.step()

                    # Generator
                    G_sample = self.G(X_tilde, M_mb)
                    Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
                    D_prob = D(Hat_X, H_mb)
                    G_loss1 = -torch.mean((1-M_mb)*torch.log(D_prob+1e-8))
                    MSE_loss = torch.sum((M_mb*(X_mb - G_sample))**2)/(torch.sum(M_mb)+1e-8)
                    G_loss = G_loss1 + self.alpha*MSE_loss
                    G_optimizer.zero_grad(); G_loss.backward(); G_optimizer.step()

            # Final imputation
            with torch.no_grad():
                Z_final = torch.rand_like(self.X_init)
                X_tilde = self.mask * self.X_init + (1 - self.mask) * Z_final
                G_sample = self.G(X_tilde, self.mask)
                X_imputed = self.X_init.clone()
                X_imputed[self.missing_mask] = G_sample[self.missing_mask]

            return X_imputed.cpu().numpy()

    def sample(self, num_samples=10):
            # Use the stored self.X_init, self.mask, self.missing_mask, self.G
            imputed_list = []
            for _ in range(num_samples):
                with torch.no_grad():
                    
                    Z_final = torch.rand_like(self.X_init)
                    X_tilde = self.mask * self.X_init + (1 - self.mask) * Z_final
                    G_sample = self.G(X_tilde, self.mask)  # keep as is
                    X_imputed = self.X_init.clone()
                    X_imputed[self.missing_mask] = G_sample[self.missing_mask]
                    imputed_list.append(X_imputed.cpu().numpy())
                    imputed_array = np.stack(imputed_list, axis=0)
            X_mean = imputed_array.mean(axis=0)
            X_std = imputed_array.std(axis=0)
            return X_mean, X_std


class GAINU:
    def __init__(self, hint_rate=0.9, alpha=100, batch_size=128, epochs=1000, lr=1e-3, device=None):
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    class Generator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim*2, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc_mu = nn.Linear(128, input_dim)
            self.fc_logvar = nn.Linear(128, input_dim)  # log variance for aleatoric uncertainty
            self.relu = nn.ReLU()
        
        def forward(self, x, m):
            inputs = torch.cat([x, m], dim=1)
            h = self.relu(self.fc1(inputs))
            h = self.relu(self.fc2(h))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim*2, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, input_dim)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x, h):
            inputs = torch.cat([x, h], dim=1)
            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            return self.sigmoid(self.fc3(x))
    
    def fit_transform_uncertainty(self, X):
        # Convert to tensor
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X = X.detach().clone().float().to(self.device)
        self.mask = (~torch.isnan(X)).float()
        self.missing_mask = torch.isnan(X)

        # Initialize missing values randomly
        self.X_init = X.clone()
        self.X_init[self.missing_mask] = torch.rand(self.missing_mask.sum()).to(self.device)

        n, d = X.shape
        self.G = self.Generator(d).to(self.device)
        D = self.Discriminator(d).to(self.device)

        G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            idx = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                batch_idx = idx[i:i+self.batch_size]
                X_mb = self.X_init[batch_idx]
                M_mb = self.mask[batch_idx]

                # ----------------------
                # Train Discriminator
                # ----------------------
                Z_mb = torch.rand_like(X_mb)
                X_tilde = M_mb * X_mb + (1 - M_mb) * Z_mb
                H_mb = M_mb * torch.bernoulli(torch.full_like(M_mb, self.hint_rate))
                with torch.no_grad():
                    G_mu, _ = self.G(X_tilde, M_mb)
                Hat_X = M_mb * X_mb + (1 - M_mb) * G_mu
                D_prob = D(Hat_X, H_mb)
                D_loss = -torch.mean(M_mb*torch.log(D_prob+1e-8) + (1-M_mb)*torch.log(1-D_prob+1e-8))
                D_optimizer.zero_grad(); D_loss.backward(); D_optimizer.step()

                # ----------------------
                # Train Generator
                # ----------------------
                G_mu, G_logvar = self.G(X_tilde, M_mb)
                Hat_X = M_mb * X_mb + (1 - M_mb) * G_mu
                D_prob = D(Hat_X, H_mb)

                # Adversarial loss
                G_loss1 = -torch.mean((1-M_mb)*torch.log(D_prob+1e-8))
                
                # NLL loss for aleatoric uncertainty
                sigma2 = torch.exp(G_logvar)
                NLL_loss = torch.sum(M_mb * ((X_mb - G_mu)**2 / (2*sigma2) + 0.5*G_logvar)) / (torch.sum(M_mb)+1e-8)

                G_loss = G_loss1 + self.alpha * NLL_loss
                G_optimizer.zero_grad(); G_loss.backward(); G_optimizer.step()

        # ----------------------
        # Final imputation with mean and std
        # ----------------------
        with torch.no_grad():
            Z_final = torch.rand_like(self.X_init)
            X_tilde = self.mask * self.X_init + (1 - self.mask) * Z_final
            G_mu, G_logvar = self.G(X_tilde, self.mask)
            X_imputed = self.X_init.clone()
            X_imputed[self.missing_mask] = G_mu[self.missing_mask]
            X_std = torch.sqrt(torch.exp(G_logvar))  # aleatoric uncertainty

        return X_imputed.cpu().numpy(), X_std.cpu().numpy()
