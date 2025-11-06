# tabular_csdi_adapter.py
# Adapter to use YOUR CSDI_model.py + diff_models.py on continuous tabular data.
# Lets you do: fit_transform(X)  and  transform_with_std(X, K)

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from CSDI_model import CSDI_base  # uses your diff_models.diff_CSDI under the hood
from Utils import nanmean, MAE, RMSE

def _mask_from_nan(X): return (~np.isnan(X)).astype(np.float32)
def _to_tensor(X): return torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)

def _std_fit(X, M):
    eps=1e-8; cnt=M.sum(0)
    mu = np.nansum(np.where(M==1, X, np.nan), 0) / np.clip(cnt,1,None)
    var = np.nansum(np.where(M==1, (X-mu)**2, np.nan), 0) / np.clip(cnt-1,1,None)
    sd = np.sqrt(np.clip(var, eps, None)); sd = np.where(sd<=1e-6, 1.0, sd)
    return mu, sd
def _std_apply(X, mu, sd): return (X - mu) / sd
def _std_inv(Z, mu, sd):   return Z * sd + mu

class TabCSDI_Tabular(CSDI_base):
    """Minimal tabular wrapper: K=1 channel, L=#features."""
    def __init__(self, config, device, target_dim: int = 1):
        super().__init__(target_dim=target_dim, config=config, device=device)

    def process_data(self, batch):
        # batch: dict with (B,L) tensors
        od = batch["observed_data"].to(self.device).float().unsqueeze(1)  # (B,1,L)
        om = batch["observed_mask"].to(self.device).float().unsqueeze(1)  # (B,1,L)
        tp = batch["timepoints"].to(self.device).float()                  # (B,L)
        gm = batch["gt_mask"].to(self.device).float().unsqueeze(1)        # (B,1,L)
        cut_length = torch.zeros(od.size(0), dtype=torch.long, device=self.device)
        for_pattern_mask = om
        return od, om, tp, gm, for_pattern_mask, cut_length

class TabCSDIImputer:
    """
    Uses YOUR CSDI implementation. API:
        imp = CSDIImputerUsingYourCode(epochs=1200, nsamples=20)
        X_imp = imp.fit_transform(X_missing)             # numpy out
        mean, std = imp.transform_with_std(X_missing,50) # posterior stats
    """
    def __init__(self,
                 epochs:int=1200, lr:float=1e-3, nsamples:int=20,
                 timeemb:int=128, featureemb:int=128,
                 num_steps:int=1000, beta_start:float=1e-4, beta_end:float=2e-2,
                 schedule:str="linear",
                 channels:int=64, layers:int=6, nheads:int=4,
                 device:Optional[str]=None, seed:int=None, verbose:bool=True):
        self.epochs=epochs; self.lr=lr; self.nsamples=nsamples
        self.verbose=verbose; self.seed=seed
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # config expected by your CSDI + diff_models
        self.config = {
            "lr": lr,
            "model": {
                "timeemb": timeemb,
                "featureemb": featureemb,
                "is_unconditional": False,
                "target_strategy": "random",
            },
            "diffusion": {
                "num_steps": num_steps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "schedule": schedule,
                "channels": channels,
                "layers": layers,
                "nheads": nheads,
            
                "diffusion_embedding_dim": 64,  # required by diff_models.diff_CSDI
                # side_dim will be set in CSDI_base.__init__
            },
        }
        self.model=None; self.mu=None; self.sd=None; self.F=None

    def _build_batch(self, X: np.ndarray):
        X = X.astype(np.float32)
        M = _mask_from_nan(X)
        Zw = X
        observed_data = _to_tensor(Zw)                       # (B,L)
        observed_mask = torch.tensor(M, dtype=torch.float32) # (B,L)
        B, L = observed_data.shape
        timepoints = torch.arange(L).unsqueeze(0).repeat(B, 1)  # (B,L)
        gt_mask = observed_mask.clone()
        batch = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "timepoints": timepoints,
            "gt_mask": gt_mask,
        }
        # move to device
        for k in batch: batch[k] = batch[k].to(self.device)
        return batch

    def fit(self, X_missing):
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)
        X = self._to_numpy(X_missing)
        assert X.ndim==2, "X must be (N,F)"
        self.F = X.shape[1]
        batch = self._build_batch(X)
        # build model
        self.model = TabCSDI_Tabular(config=self.config, device=self.device, target_dim=1).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for e in range(1, self.epochs+1):
            loss = self.model(batch, is_train=1)
            opt.zero_grad(); loss.backward(); opt.step()
            if self.verbose and (e==1 or e%max(1,self.epochs//10)==0 or e==self.epochs):
                print(f"[{e:4d}/{self.epochs}] loss={loss.item():.6f}")
        return self

    @torch.no_grad()
    def transform(self, X_missing):
        mean, _ = self.transform_with_std(X_missing, self.nsamples)
        return mean
    
    def fit_transform(self, X_missing):
        self.fit(X_missing)
        return self.transform(X_missing)


    @torch.no_grad()
    def transform_with_std(self, X_missing, num_samples:int=50):
        assert self.model is not None, "Call fit() first."
        X = self._to_numpy(X_missing)
        assert X.shape[1]==self.F, "Feature dimension mismatch."
        batch = self._build_batch(X)
        self.model.eval()
        samples, observed_data, target_mask, observed_mask, _ = self.model.evaluate(batch, n_samples=num_samples)
        imputed_mean = samples.mean(dim=1).squeeze(1)  # (B,L)
        imputed_std  = samples.std (dim=1, unbiased=False).squeeze(1)  # (B,L)
        mean_np = imputed_mean.cpu().numpy()
        std_np  = imputed_std.cpu().numpy()
        
        # keep observed values and set std=NaN on observed
        observed = ~np.isnan(X)
        mean_np = np.where(observed, X, mean_np)
        std_np  = np.where(observed, np.nan, std_np)
        return mean_np, std_np

    @staticmethod
    def _to_numpy(X):
        if isinstance(X, np.ndarray): return X
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame): return X.values
        except Exception: pass
        if torch.is_tensor(X): return X.detach().cpu().numpy()
        raise TypeError("X must be numpy array, pandas DataFrame, or torch Tensor")
