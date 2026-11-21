"""Window-level estimators.

Tabular families (constant, calibration, logistic, ridge, mlp) need only NumPy,
pandas, scikit-learn and joblib. ``family='cnn'`` is classification-only: it
reads a log-mel window field, convolves the 32x22 map, and stores its weights as
NumPy. Predictions therefore always run on CPU, including after a CUDA training
run and after a joblib reload on a machine without CUDA.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

BANDS = [f'f{i}' for i in range(32)]
AUDIO = BANDS + ['b4_8', 'rms_db']
# log-mel window: 32 bands x 22 frames, flattened C order, absolute dB.
LOGMEL_SHAPE = (32, 22)
LOGMEL_SIZE = LOGMEL_SHAPE[0] * LOGMEL_SHAPE[1]
BATCH = 512


def weights(d):
    w = 1. / d.groupby('sample_id')['sample_id'].transform('size').to_numpy(float)
    return w  # unit mass per take for fitting and evaluation


def logmel_array(d):
    """Stack the log-mel field to (windows, 32 bands, 22 frames), C order, absolute dB."""
    if 'logmel' not in d:
        raise ValueError("family='cnn' requires the logmel column")
    raw = d['logmel'].to_numpy()
    if raw.dtype == object:
        arr = np.stack([np.asarray(v, dtype=float).ravel() for v in raw])
    else:
        arr = np.asarray(raw, dtype=float).reshape(len(d), -1)
    if arr.shape[1] != LOGMEL_SIZE:
        raise ValueError(f'logmel must hold {LOGMEL_SIZE} values per window')
    if not np.isfinite(arr).all():
        raise ValueError('non-finite logmel values')
    return arr.reshape(len(d), *LOGMEL_SHAPE)


def band_standardisation(x, w):
    """Per-band weighted mean/std over training windows and frames only."""
    wn = w / w.sum()
    mu = (wn[:, None] * x.mean(axis=2)).sum(axis=0)
    var = (wn[:, None] * ((x - mu[None, :, None]) ** 2).mean(axis=2)).sum(axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return mu, np.where(sd < 1e-6, 1.0, sd)


class Model:
    def __init__(self, task, params):
        self.task, self.p = task, params.copy()

    def _x(self, d):
        bands = self.p.get('bands', 'full')
        # Use fixed contiguous mel bands for low/mid ablations; 4–8 kHz is exact.
        edges = 700 * np.expm1(np.linspace(0, np.log1p(12000 / 700), 33))
        centers = (edges[:-1]+edges[1:])/2
        cols = {'full': AUDIO, '4-8k': ['b4_8'], 'rms': ['rms_db'],
                'low_mel': [c for c,h in zip(BANDS,centers) if 200<=h<1000],
                'mid_mel': [c for c,h in zip(BANDS,centers) if 1000<=h<4000]}[bands]
        x = d[cols].to_numpy(float, copy=True)
        if self.p.get('relative'):
            # Log power relative to the broadband level; drop the all-zero RMS channel.
            x -= d[['rms_db']].to_numpy()
            if bands == 'full': x = x[:, :-1]
        devices = d.device_id.astype(str).to_numpy()
        if self.p['family'] == 'calibration':
            x[:, 0] -= np.array([self.refs.get(dev, self.ref_default) for dev in devices])
        if self.p.get('phone'):
            x = np.column_stack([x] + [devices == dev for dev in self.devices])
        return x

    def _dropout(self):
        drop = float(self.p.get('dropout', 0.0))
        if not 0.0 <= drop <= 0.5:
            raise ValueError('dropout must be within [0, 0.5]')
        return drop

    def _check_cnn(self):
        if self.task != 'classification':
            raise ValueError("family='cnn' is classification-only")
        if self.p.get('bands', 'full') != 'full':
            raise ValueError("family='cnn' uses the full log-mel map: bands must be 'full'")
        if self.p.get('phone'):
            raise ValueError("family='cnn' does not support the phone one-hot (phone=False)")
        if self.p.get('width') not in (8, 16, 32):
            raise ValueError('cnn width must be one of 8, 16, 32')
        if int(self.p.get('depth', 0)) not in (1, 2):
            raise ValueError('cnn depth must be 1 or 2')
        self._dropout()

    def _cnn_input(self, d):
        x = logmel_array(d)
        if self.p.get('relative'):
            # Remove the broadband level; keep the time axis and the band shape.
            x = x - d['rms_db'].to_numpy(float)[:, None, None]
        return x

    def _cnn_net(self):
        from torch import nn
        width, depth, drop = int(self.p['width']), int(self.p['depth']), self._dropout()
        layers, channels = [], 1
        for _ in range(depth):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU()])
            if drop:
                layers.append(nn.Dropout2d(drop))
            channels = width
        # Keep a coarse 2x2 band/time map instead of collapsing to one vector.
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(width * 4, 1))

    @staticmethod
    def _cnn_sums(net, tx, ty, tw, batch, criterion):
        """Weighted loss/accuracy sums over batches; no dropout at eval."""
        import torch
        net.eval(); loss_sum = weight_sum = correct_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(tx), batch):
                stop = start + batch
                logits = net(tx[start:stop].unsqueeze(1)).flatten()
                loss = criterion(logits, ty[start:stop])
                wb = tw[start:stop]
                loss_sum += float((loss * wb).sum())
                weight_sum += float(wb.sum())
                correct_sum += float((((logits >= 0) == ty[start:stop].bool()).float() * wb).sum())
        return loss_sum, weight_sum, correct_sum

    def _fit_cnn(self, d, valid, device, epochs, patience, seed):
        import torch
        from torch import nn
        self._check_cnn()
        if valid is not None and len(valid) == 0:
            valid = None
        w = weights(d)
        x = self._cnn_input(d)
        self.lm_mean, self.lm_std = band_standardisation(x, w)
        tx = self._scale_cnn(x)
        ty = d.drone_present.to_numpy(np.float32)
        # Same convention as the MLP path: global train mean weight 1.
        tw = w / w.mean()
        if valid is not None:
            vx = self._cnn_input(valid)
            vtx = self._scale_cnn(vx)
            vty = valid.drone_present.to_numpy(np.float32)
            vtw = weights(valid)

        torch.manual_seed(seed)
        net = self._cnn_net().to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=float(self.p['lr']), weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        t_tx = torch.as_tensor(tx, dtype=torch.float32, device=device)
        t_ty = torch.as_tensor(ty, dtype=torch.float32, device=device)
        t_tw = torch.as_tensor(tw, dtype=torch.float32, device=device)
        if valid is not None:
            t_vx = torch.as_tensor(vtx, dtype=torch.float32, device=device)
            t_vy = torch.as_tensor(vty, dtype=torch.float32, device=device)
            t_vw = torch.as_tensor(vtw, dtype=torch.float32, device=device)
        batch = int(self.p.get('batch_size', BATCH))
        if batch < 1:
            raise ValueError('batch_size must be positive')
        generator = torch.Generator().manual_seed(seed)
        best, self.best_epoch, self.cnn_state = np.inf, 0, None
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(len(t_tx), generator=generator)
            for start in range(0, len(t_tx), batch):
                idx = order[start:start + batch]
                optim.zero_grad(set_to_none=True)
                logits = net(t_tx[idx].unsqueeze(1)).flatten()
                loss = (criterion(logits, t_ty[idx]) * t_tw[idx]).mean()
                loss.backward()
                optim.step()
            loss_sum, _, correct_sum = self._cnn_sums(net, t_tx, t_ty, t_tw, batch, criterion)
            train_loss, train_accuracy = loss_sum / len(t_tx), correct_sum / len(t_tx)
            row = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy}
            if valid is None:
                score = train_loss
            else:
                v_loss, v_weight, v_correct = self._cnn_sums(net, t_vx, t_vy, t_vw, batch, criterion)
                score = v_loss / v_weight if v_weight else float('nan')
                row.update(val_loss=score, val_accuracy=v_correct / v_weight if v_weight else 0.0)
            self.curve.append(row)
            if not np.isfinite(score):
                raise ValueError('Non-finite network loss')
            # No validation fold: final refit runs exactly `epochs` and keeps the last one, as the MLP.
            if valid is None or score < best - 1e-6:
                best, self.best_epoch = score, epoch
                self.cnn_state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
            elif epoch - self.best_epoch >= patience:
                break
        if self.cnn_state is None:
            raise ValueError('cnn produced no finite checkpoint')
        return self

    def _scale_cnn(self, x):
        return ((x - self.lm_mean[None, :, None]) / self.lm_std[None, :, None]).astype(np.float32)

    def _predict_cnn(self, d):
        """Deterministic CPU inference from stored NumPy weights, one batch at a time."""
        import torch
        net = self._cnn_net()
        net.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in self.cnn_state.items()})
        net.eval()
        x = self._scale_cnn(self._cnn_input(d))
        batch = int(self.p.get('batch_size', BATCH))
        out = []
        with torch.inference_mode():
            for start in range(0, len(x), batch):
                logits = net(torch.from_numpy(x[start:start + batch]).unsqueeze(1)).flatten()
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    def fit(self, d, valid=None, device='cpu', epochs=60, patience=8, seed=42):
        family = self.p['family']; cls = self.task == 'classification'
        self.devices = sorted(d.device_id.astype(str).unique()); self.curve = []
        if family == 'constant': return self
        if family == 'cnn': return self._fit_cnn(d, valid, device, epochs, patience, seed)
        if family == 'calibration':
            # Median levels of independent 1 m takes, not of individual windows.
            anchors = d.loc[np.isclose(d.distance_m, 1)].groupby(['device_id','sample_id']).b4_8.median()
            self.refs = {str(k): float(v) for k,v in anchors.groupby(level=0).median().items()}
            self.ref_default = float(anchors.median()) if len(anchors) else 0.
        y = d.drone_present.to_numpy(int) if cls else d.distance_m.to_numpy(float)
        w = weights(d); x = self._x(d)
        self.scaler = StandardScaler().fit(x, sample_weight=w); x = self.scaler.transform(x)
        if family != 'mlp':
            reg = self.p.get('strength', 1.)
            self.estimator = LogisticRegression(C=reg, max_iter=1000) if cls else Ridge(alpha=reg)
            self.estimator.fit(x, np.log(y) if family == 'calibration' else y, sample_weight=w)
            return self
        import torch
        from torch import nn
        drop = self._dropout()
        torch.manual_seed(seed)
        layers = []; width = self.p['width']; dim = x.shape[1]
        for _ in range(self.p['depth']):
            layers.append(nn.Linear(dim, width))
            if drop: layers.append(nn.Dropout(drop))
            layers.append(nn.ReLU())
            dim = width
        net = nn.Sequential(*layers, nn.Linear(dim, 1)).to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=self.p['lr'])
        def tensor(a): return torch.as_tensor(a, dtype=torch.float32, device=device)
        tx, ty, tw = map(tensor, (x, y, w / w.mean()))
        criterion = nn.BCEWithLogitsLoss(reduction='none') if cls else nn.SmoothL1Loss(reduction='none')
        if valid is not None:
            vx = tensor(self.scaler.transform(self._x(valid)))
            vy = tensor(valid.drone_present.to_numpy(float) if cls else valid.distance_m.to_numpy(float)); vw = tensor(weights(valid))
        best = np.inf; self.best_epoch = 0
        for epoch in range(1, epochs+1):
            net.train()
            for idx in torch.randperm(len(y), device=device).split(128):
                optim.zero_grad(); loss = (criterion(net(tx[idx]).flatten(), ty[idx])*tw[idx]).mean(); loss.backward(); optim.step()
            net.eval()
            with torch.no_grad():
                tp = net(tx).flatten()
                train = float((criterion(tp, ty)*tw).mean())
                row = {'epoch':epoch,'train_loss':train}
                row['train_accuracy' if cls else 'train_mae'] = float((((tp>=0)==ty if cls else abs(tp-ty))*tw).mean())
                if valid is not None:
                    pred = net(vx).flatten(); score = float((criterion(pred, vy)*vw).sum()/vw.sum()); row['val_loss'] = score
                    row['val_accuracy' if cls else 'val_mae'] = float((((pred>=0)==vy if cls else abs(pred-vy))*vw).sum()/vw.sum())
                else: score = train
            self.curve.append(row)
            if not np.isfinite(score): raise ValueError('Non-finite network loss')
            if valid is None or score < best-1e-6:
                best = score; self.best_epoch = epoch
                self.layers = [(l.weight.detach().cpu().numpy().copy(), l.bias.detach().cpu().numpy().copy()) for l in net if isinstance(l, nn.Linear)]
            elif epoch-self.best_epoch >= patience: break
        return self

    def predict(self, d, *, bounded=True):
        """DataFrame -> probability of presence or estimated metres; one value per row."""
        family = self.p['family']
        if family == 'constant': return np.full(len(d), 3.)
        if family == 'cnn': return self._predict_cnn(d)
        x = self.scaler.transform(self._x(d))
        if family == 'mlp':
            for i, (w,b) in enumerate(self.layers):
                x = x @ w.T + b
                if i < len(self.layers)-1: x = np.maximum(0,x)
            p = x.ravel()
            if self.task == 'classification': return 1/(1+np.exp(-np.clip(p,-60,60)))
        elif self.task == 'classification': return self.estimator.predict_proba(x)[:,1]
        else: p = self.estimator.predict(x)
        if family == 'calibration': p = np.exp(np.clip(p,-30,30))
        return np.clip(p, 1., 5.) if bounded else p

    def parameter_count(self):
        """Number of fitted scalars for every family, including the CNN state."""
        family = self.p['family']
        if family == 'constant': return 0
        if family == 'mlp': return int(sum(w.size + b.size for w, b in self.layers))
        if family == 'cnn': return int(sum(v.size for v in self.cnn_state.values()))
        return int(np.size(self.estimator.coef_) + np.size(self.estimator.intercept_))
