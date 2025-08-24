import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

class MIWAEImputer:
    def __init__(self, input_dim, latent_dim=5, hidden_dims=[128, 64],
                 K=5, lr=1e-3, epochs=100, batch_size=64, device=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.K = K
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model(hidden_dims)
        self._init_weights()

    def _build_model(self, hidden_dims):
    # ---------------- Encoder ----------------
        layers = []
        in_dim = self.input_dim * 2  # input = x*mask + mask
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers).to(self.device)
        self.enc_mean = nn.Linear(in_dim, self.latent_dim).to(self.device)
        self.enc_logvar = nn.Linear(in_dim, self.latent_dim).to(self.device)

        # ---------------- Decoder ----------------
        layers = []
        in_dim = self.latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        # Output both mean and log-variance for full Gaussian likelihood
        layers.append(nn.Linear(in_dim, self.input_dim * 2))
        self.decoder = nn.Sequential(*layers).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.enc_mean.parameters()) +
            list(self.enc_logvar.parameters()) +
            list(self.decoder.parameters()),
            lr=self.lr
        )


    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        self.encoder.apply(init_fn)
        self.enc_mean.apply(init_fn)
        self.enc_logvar.apply(init_fn)
        self.decoder.apply(init_fn)

    def _loss(self, x_obs, mask):
        eps_small = 1e-6
        batch_size = x_obs.size(0)

        # -------- Encode --------
        h = torch.cat([x_obs * mask, mask], dim=1)
        h = self.encoder(h)
        mu = self.enc_mean(h)                 # (B, L)
        logvar = self.enc_logvar(h)           # (B, L)
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        std = torch.exp(0.5 * logvar) + eps_small

        # -------- Sample K latent variables --------
        mu = mu.unsqueeze(1).expand(batch_size, self.K, -1)          # (B, K, L)
        std = std.unsqueeze(1).expand(batch_size, self.K, -1)       # (B, K, L)
        eps = torch.randn_like(std)
        z = mu + eps * std                                           # (B, K, L)

        # -------- Decode --------
        z_flat = z.view(batch_size * self.K, -1)
        x_dec = self.decoder(z_flat).view(batch_size, self.K, self.input_dim * 2)
        x_mu = x_dec[:, :, :self.input_dim]
        x_logvar = x_dec[:, :, self.input_dim:]
        x_std = torch.exp(0.5 * x_logvar) + eps_small

        # -------- Reconstruction log-likelihood (full Gaussian) --------
        recon_term = -0.5 * (((x_obs.unsqueeze(1) - x_mu) ** 2) / (x_std ** 2) 
                            + 2 * torch.log(x_std) + np.log(2 * np.pi))
        recon_term = (recon_term * mask.unsqueeze(1)).sum(-1)  # sum only over observed entries

        # -------- Prior log-prob --------
        prior_log = -0.5 * (z ** 2).sum(-1)

        # -------- Posterior log-prob --------
        log_qzx = -0.5 * (((z - mu) ** 2) / (std ** 2) + 2 * torch.log(std) + np.log(2 * np.pi)).sum(-1)

        # -------- IWAE log-weight --------
        log_w = recon_term + prior_log - log_qzx
        log_sum = torch.logsumexp(log_w, dim=1) - np.log(self.K)
        loss = - torch.mean(log_sum)

        return loss, x_mu, (mu, logvar)


    def fit_transform(self, X_missing):
        """
        Fit MIWAE model on X_missing and return imputed values
        """
        # Convert to numpy
        if hasattr(X_missing, "values"):
            X_arr = X_missing.values.astype(np.float32)
        else:
            X_arr = np.asarray(X_missing).astype(np.float32)

        # Mask: 1 = observed, 0 = missing
        mask_np = (~np.isnan(X_arr)).astype(np.float32)

        # Temporary fill missing with 0
        X_fill = X_arr.copy()
        X_fill[mask_np == 0] = 0.0

        # Convert to tensor
        X = torch.tensor(X_fill, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X, mask)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # ===== Training loop =====
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, mb in loader:
                xb = xb.to(self.device)
                mb = mb.to(self.device)
                self.optimizer.zero_grad()
                loss, _, _ = self._loss(xb, mb)
                if torch.isnan(loss) or torch.isinf(loss):
                    return np.full_like(X_arr, np.nan)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
          

        # ===== Imputation =====
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            N = X.shape[0]

            # Encoder outputs
            h_all = self.encoder(torch.cat([X * mask, mask], dim=1))
            mu_all = self.enc_mean(h_all)
            logvar_all = self.enc_logvar(h_all)
            logvar_all = torch.clamp(logvar_all, min=-10.0, max=5.0)

            # Repeat for K samples
            mu = mu_all.unsqueeze(1).repeat(1, self.K, 1)        # (N, K, latent_dim)
            logvar = logvar_all.unsqueeze(1).repeat(1, self.K, 1) # (N, K, latent_dim)
            std = torch.exp(0.5 * logvar) + 1e-6
            eps = torch.randn_like(std)
            z = mu + eps * std  # (N, K, latent_dim)

            # Decode
            x_dec = self.decoder(z.view(N * self.K, self.latent_dim))  # (N*K, 2*D)
            x_mu = x_dec[:, :self.input_dim].view(N, self.K, self.input_dim)
            # x_logvar = x_dec[:, self.input_dim:].view(N, self.K, self.input_dim)  # optional

            # Mean over K samples
            x_mean = x_mu.mean(dim=1)  # (N, D)

            # Impute missing entries only
            X_completed = mask * X + (1.0 - mask) * x_mean

        return X_completed.cpu().numpy()


    def fit_transform_std(self, X_missing, return_std=True, inference_K=None):
        """
        Fit MIWAE and return imputed values and aleatoric uncertainty.
        """
        if hasattr(X_missing, "values"):
            X_arr = X_missing.values.astype(np.float32)
        else:
            X_arr = np.asarray(X_missing).astype(np.float32)

        mask_np = (~np.isnan(X_arr)).astype(np.float32)
        X_fill = X_arr.copy()
        X_fill[mask_np == 0] = 0.0

        X = torch.tensor(X_fill, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X, mask)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # ===== Training =====
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, mb in loader:
                xb = xb.to(self.device)
                mb = mb.to(self.device)
                self.optimizer.zero_grad()
                loss, _, _ = self._loss(xb, mb)
                if torch.isnan(loss) or torch.isinf(loss):
                    print("[Warning] NaN/Inf loss encountered. Stopping training early.")
                    return np.full_like(X_arr, np.nan)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
           

        # ===== Inference =====
        inference_K = inference_K or self.K
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            N = X.shape[0]

            h_all = self.encoder(torch.cat([X * mask, mask], dim=1))
            mu_all = self.enc_mean(h_all)
            logvar_all = self.enc_logvar(h_all)
            logvar_all = torch.clamp(logvar_all, min=-10.0, max=5.0)

            mu = mu_all.unsqueeze(1).repeat(1, inference_K, 1)
            logvar = logvar_all.unsqueeze(1).repeat(1, inference_K, 1)
            std = torch.exp(0.5 * logvar) + 1e-6
            eps = torch.randn_like(std)
            z = mu + eps * std

            # Decode
            x_dec = self.decoder(z.view(N * inference_K, self.latent_dim))
            x_mu = x_dec[:, :self.input_dim].view(N, inference_K, self.input_dim)
            x_logvar = x_dec[:, self.input_dim:].view(N, inference_K, self.input_dim)
            x_std = torch.exp(0.5 * x_logvar)

            # Mean and std
            x_mean = x_mu.mean(dim=1)
            x_std_mean = x_std.mean(dim=1)

            # Impute missing only
            X_completed = mask * X + (1.0 - mask) * x_mean

            # Mask std for observed entries
            x_std_masked = x_std_mean.cpu().numpy()
            x_std_masked[mask.cpu().numpy() == 1] = np.nan

        if return_std:
            return X_completed.cpu().numpy(), x_std_masked
        else:
            return X_completed.cpu().numpy()
