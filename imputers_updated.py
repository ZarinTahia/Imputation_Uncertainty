#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from Utils import nanmean, MAE, RMSE
import logging

class RRimputer_nll():
    def __init__(self,
                 models, 
                 eps=0.01, 
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 max_iter=5,
                 niter=5, 
                 batchsize=128,
                 n_pairs=2, 
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5, 
                 order='random',
                 unsymmetrize=True, 
                 scaling=.9,
                 var_reg_weight=0.001,  # Reduced from 0.01
                 sample=False):

        self.models = models
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="auto")
        self.lr = lr
        self.opt = opt
        self.max_iter = max_iter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.var_reg_weight = var_reg_weight
        self.is_fitted = False
        self.sample = sample

    def gaussian_nll_loss(self, mean, log_var, target):
        var = torch.exp(log_var).clamp(min=1e-4, max=1e4)
        return 0.5 * (torch.log(var) + 0.5 * ((target - mean)**2 / var))

    def fit_transform(self, X, verbose=True, report_interval=1, X_true=None):
        X = X.clone()
        n, d = X.shape
        mask = torch.isnan(X).double()
        variances = torch.zeros_like(X)
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        order_ = torch.argsort(mask.sum(0))
        optimizers = [self.opt(self.models[i].parameters(), lr=self.lr, weight_decay=self.weight_decay) 
                      for i in range(d)]

        # Initialize missing values
        imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
        X[mask.bool()] = imps
        X_filled = X.clone()

        if X_true is not None:
            maes = np.zeros(self.max_iter)
            rmses = np.zeros(self.max_iter)

        self.cell_uncertainty_log = {j: [] for j in range(d)}

        for i in range(self.max_iter):
            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):
                j = order_[l].item()
                if (~mask[:, j].bool()).sum().item() == n:
                    continue  # No missing values

                for k in range(self.niter):
                    # Current predictions
                    mean, log_var = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1:d]])
                    log_var = torch.clamp(log_var, min=-4.0, max=4.0)  # More stable bounds

                    # Get targets - use current imputation values (detached)
                    with torch.no_grad():
                        target = X_filled[mask[:, j].bool(), j].clone()

                    # Calculate losses
                    nll_loss = self.gaussian_nll_loss(mean, log_var, target).mean()
                    
                    # Sinkhorn loss
                    sinkhorn_loss = 0
                    for _ in range(self.n_pairs):
                        idx1 = np.random.choice(n, self.batchsize, replace=False)
                        X1 = X_filled[idx1]
                        
                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batchsize, replace=self.batchsize > n_miss)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]
                        else:
                            idx2 = np.random.choice(n, self.batchsize, replace=False)
                            X2 = X_filled[idx2]
                        
                        sinkhorn_loss += self.sk(X1, X2)

                    # Combined loss with gentle variance regularization
                    loss = (sinkhorn_loss / self.n_pairs + 
                            nll_loss + 
                            self.var_reg_weight * torch.mean(torch.exp(-log_var)))

                    # Optimize
                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                    # Update imputation
                    with torch.no_grad():
                        if self.sample:
                            var = torch.exp(log_var).clamp(min=1e-4, max=1e4)
                            std = torch.sqrt(var)
                            epsilon = torch.randn_like(mean)
                            X_filled[mask[:, j].bool(), j] = (mean + std * epsilon).squeeze()
                        else:
                            X_filled[mask[:, j].bool(), j] = mean.squeeze()

                    # Log uncertainty
                    self.cell_uncertainty_log[j].append(log_var.detach().cpu().numpy().squeeze())

                # Final imputation for this feature in this cycle
                with torch.no_grad():
                    mean, log_var = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1:d]])
                    X_filled[mask[:, j].bool(), j] = mean.squeeze()
                    variances[mask[:, j].bool(), j] = torch.exp(log_var.squeeze())

            # Validation and logging
            if X_true is not None:
                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()

            if verbose and (i % report_interval == 0):
                if X_true is not None:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item():.4f}\t'
                                f'Validation MAE: {maes[i]:.4f}\t'
                                f'RMSE: {rmses[i]:.4f}')
                else:
                    logging.info(f'Iteration {i}:\t Loss: {loss.item():.4f}')

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        self.is_fitted = True
        variances = torch.clamp(variances, min=1e-6, max=1e4)
        
        if X_true is not None:
            return X_filled, maes, rmses, variances, self.cell_uncertainty_log
        else:
            return X_filled, variances, self.cell_uncertainty_log

    def transform(self, X, mask, verbose=True, report_interval=1, X_true=None):
        assert self.is_fitted, "The model has not been fitted yet."
        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
        
        X = X.clone()
        X[mask.bool()] = nanmean(X)
        X_filled = X.clone()
        variances = torch.zeros_like(X)

        for i in range(self.max_iter):
            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for j in order_:
                j = j.item()
                with torch.no_grad():
                    mean, log_var = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1:d]])
                    X_filled[mask[:, j].bool(), j] = mean.squeeze()
                    variances[mask[:, j].bool(), j] = torch.exp(log_var.squeeze())

            if verbose and (i % report_interval == 0) and (X_true is not None):
                logging.info(f'Iteration {i}:\t '
                            f'Validation MAE: {MAE(X_filled, X_true, mask).item():.4f}\t'
                            f'RMSE: {RMSE(X_filled, X_true, mask).item():.4f}')

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')
        
        variances = torch.clamp(variances, min=1e-6, max=1e4)
        return X_filled, variances