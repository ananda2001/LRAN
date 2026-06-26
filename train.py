import torch
import torch.nn as nn


def train(model, train_loader, num_epochs, lr, wd, gradclip,
          gamma_id, gamma_fwd, gamma_lin, gamma_eig=0.0, device='cpu', print_every=50):

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()

    history = {'loss': [], 'loss_id': [], 'loss_fwd': [], 'loss_lin': [], 'loss_eig': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        totals = {k: 0.0 for k in history}

        for batch in train_loader:
            # batch = [x_0, x_1, ..., x_K, u_0, ..., u_{K-1}]
            K  = (len(batch) - 1) // 2
            xs = [b.to(device) for b in batch[:K + 1]]
            us = torch.stack([b.to(device) for b in batch[K + 1:]], dim=1)  # (B, K, n_u)

            # Encode all K+1 states once; reused for both L_id and L_lin
            z_enc = [model.encoder(xs[k]) for k in range(K + 1)]

            # L_id: encode -> decode every state in the window
            loss_id = sum(mse(model.decoder(z_enc[k]), xs[k])
                          for k in range(K + 1)) / (K + 1)

            # Latent rollout from z_0
            z_preds = model.rollout(z_enc[0], us)

            # L_lin: predicted latent vs encoded ground truth
            loss_lin = sum(mse(z_preds[k], z_enc[k + 1])
                           for k in range(K)) / K

            # L_fwd: decoded prediction vs ground truth state
            loss_fwd = sum(mse(model.decoder(z_preds[k]), xs[k + 1])
                           for k in range(K)) / K

            # L_eig: penalize eigenvalues of A outside the unit circle
            if gamma_eig > 0:
                eigs    = torch.linalg.eigvals(model.A.weight)
                loss_eig = (eigs.abs() - 1).clamp(min=0).sum()
            else:
                loss_eig = torch.zeros(1, device=device)

            loss = (gamma_id  * loss_id  +
                    gamma_fwd * loss_fwd +
                    gamma_lin * loss_lin +
                    gamma_eig * loss_eig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
            optimizer.step()

            totals['loss']     += loss.item()
            totals['loss_id']  += loss_id.item()
            totals['loss_fwd'] += loss_fwd.item()
            totals['loss_lin'] += loss_lin.item()
            totals['loss_eig'] += loss_eig.item()

        n = len(train_loader)
        for k in totals:
            totals[k] /= n
            history[k].append(totals[k])

        if epoch % print_every == 0:
            print(f'Epoch {epoch:4d} | '
                  f'loss {totals["loss"]:.4e}  '
                  f'id {totals["loss_id"]:.4e}  '
                  f'fwd {totals["loss_fwd"]:.4e}  '
                  f'lin {totals["loss_lin"]:.4e}  '
                  f'eig {totals["loss_eig"]:.4e}')

    return history
