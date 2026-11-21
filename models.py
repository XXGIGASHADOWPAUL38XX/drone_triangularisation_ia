"""Modèles gelés d'après l'HPO préliminaire, plus les baselines de comparaison.

Architectures retenues par l'Optuna du run RTX (à ne re-tuner que sciemment) :
  - Détection : CNN 2D sur log-mel *relative* (niveau global de la fenêtre
    retiré), un bloc conv 32 canaux, dropout 0.15, ~450 paramètres, seuil 0.5,
    sans one-hot téléphone.
  - Distance  : MLP une couche cachée de 16 unités, dropout 0.15, sur les
    niveaux absolus pleine bande, sans one-hot téléphone, sortie bornée
    [1, 5] m.

Baselines de contexte : logistique (détection), ridge et constant 3 m
(distance). Les poids d'apprentissage et d'évaluation donnent une masse
totale égale à chaque prise, jamais à chaque fenêtre.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from features import LOGMEL, N_BANDS

BATCH_CNN = 512
BATCH_MLP = 128


def weights(d):
    """Une masse égale par prise, répartie sur ses fenêtres."""
    return 1. / d.groupby('sample_id')['sample_id'].transform('size').to_numpy(float)


def logmel_array(d):
    """Colonne logmel -> (fenêtres, 32, trames), dB absolus."""
    if LOGMEL not in d:
        raise ValueError("ce modèle nécessite la colonne 'logmel' du cache")
    raw = d[LOGMEL].to_numpy()
    if raw.dtype == object:
        arr = np.stack([np.asarray(v, dtype=float).ravel() for v in raw])
    else:
        arr = np.asarray(raw, dtype=float).reshape(len(d), -1)
    if arr.shape[1] % N_BANDS or not np.isfinite(arr).all():
        raise ValueError(f'logmel invalide (attendu un multiple de {N_BANDS} valeurs finies)')
    return arr.reshape(len(d), N_BANDS, -1)


def _dropout_ok(drop):
    if not 0.0 <= float(drop) <= 0.5:
        raise ValueError('dropout doit être dans [0, 0.5]')
    return float(drop)


def _flat_x(d, cols, relative):
    x = d[cols].to_numpy(float, copy=True)
    if relative:
        if 'rms_db' not in d:
            raise ValueError("l'option relative nécessite la colonne 'rms_db'")
        x -= d[['rms_db']].to_numpy()
        x = np.delete(x, [i for i, c in enumerate(cols) if c == 'rms_db'], axis=1)
    return x


def _band_stats(x, w):
    """Moyenne/écart-type par canal mel, pondérés par prise, train uniquement."""
    wn = w / w.sum()
    mu = (wn[:, None] * x.mean(axis=2)).sum(axis=0)
    var = (wn[:, None] * ((x - mu[None, :, None]) ** 2).mean(axis=2)).sum(axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return mu, np.where(sd < 1e-6, 1.0, sd)


class CnnPresence:
    """Détection : CNN convolutif figé sur log-mel relative (sans métadonnées)."""
    task = 'classification'

    def __init__(self, lr=7.7e-3, dropout=0.15, channels=32, relative=True):
        self.lr, self.dropout, self.channels, self.relative = float(lr), _dropout_ok(dropout), int(channels), bool(relative)
        self.curve, self.best_epoch, self.state = [], 0, None

    def _input(self, d):
        x = logmel_array(d)
        if self.relative:
            x = x - d['rms_db'].to_numpy(float)[:, None, None]
        return x

    def _net(self):
        import torch
        from torch import nn
        c = self.channels
        return nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(self.dropout) if self.dropout else nn.Identity(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(c * 4, 1),
        )

    def fit(self, d, valid=None, device='cpu', epochs=60, patience=8, seed=42, cols=None):
        import torch
        from torch import nn
        if valid is not None and len(valid) == 0:
            valid = None
        w = weights(d)
        x = self._input(d)
        self.mu, self.sd = _band_stats(x, w)
        tx = torch.as_tensor(self._scale(x), dtype=torch.float32, device=device)
        ty = torch.as_tensor(d.drone_present.to_numpy(np.float32), device=device)
        tw = torch.as_tensor(w / w.mean(), dtype=torch.float32, device=device)
        if valid is not None:
            vx = torch.as_tensor(self._scale(self._input(valid)), dtype=torch.float32, device=device)
            vy = torch.as_tensor(valid.drone_present.to_numpy(np.float32), device=device)
            vw = torch.as_tensor(weights(valid), dtype=torch.float32, device=device)
        torch.manual_seed(seed)
        net = self._net().to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        batch = BATCH_CNN
        gen = torch.Generator().manual_seed(seed)
        best = np.inf
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(len(tx), generator=gen)
            for start in range(0, len(tx), batch):
                idx = order[start:start + batch]
                optim.zero_grad(set_to_none=True)
                logits = net(tx[idx].unsqueeze(1)).flatten()
                loss = (criterion(logits, ty[idx]) * tw[idx]).mean()
                loss.backward()
                optim.step()
            net.eval()
            with torch.inference_mode():
                tp = net(tx.unsqueeze(1)).flatten()
                train_loss = float((criterion(tp, ty) * tw).mean())
                row = {'epoch': epoch, 'train_loss': train_loss,
                       'train_accuracy': float((((tp >= 0) == ty.bool()).float() * tw).mean())}
                if valid is None:
                    score = train_loss
                else:
                    vp = net(vx.unsqueeze(1)).flatten()
                    score = float((criterion(vp, vy) * vw).sum() / vw.sum())
                    row['val_loss'] = score
                    row['val_accuracy'] = float((((vp >= 0) == vy.bool()).float() * vw).sum() / vw.sum())
            self.curve.append(row)
            if not np.isfinite(score):
                raise ValueError('loss non finie')
            # Sans fold de validation : on exécute exactement `epochs` époques.
            if valid is None or score < best - 1e-6:
                best, self.best_epoch = score, epoch
                self.state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
            elif epoch - self.best_epoch >= patience:
                break
        if self.state is None:
            raise ValueError('aucun point de contrôle fini')
        return self

    def _scale(self, x):
        return ((x - self.mu[None, :, None]) / self.sd[None, :, None]).astype(np.float32)

    def predict(self, d):
        """Probabilité de présence par fenêtre, inférence CPU déterministe."""
        import torch
        net = self._net()
        net.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in self.state.items()})
        net.eval()
        x = self._scale(self._input(d))
        out = []
        with torch.inference_mode():
            for start in range(0, len(x), BATCH_CNN):
                logits = net(torch.from_numpy(x[start:start + BATCH_CNN]).unsqueeze(1)).flatten()
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=float)

    def parameter_count(self):
        return int(sum(v.size for v in self.state.values()))


class Mlp:
    """Petit MLP pour l'une ou l'autre tâche (gelé : 16 cachées, dropout 0.15)."""
    def __init__(self, task, width=16, depth=1, dropout=0.15, lr=3e-3, relative=False):
        self.task, self.width, self.depth = task, int(width), int(depth)
        self.lr, self.dropout, self.relative = float(lr), _dropout_ok(dropout), bool(relative)
        self.curve, self.best_epoch, self.layers = [], 0, None

    def fit(self, d, valid=None, device='cpu', epochs=60, patience=8, seed=42, cols=None):
        import torch
        from torch import nn
        if cols is None:
            raise ValueError('préciser les colonnes de features')
        self.cols = list(cols)
        w = weights(d)
        x = _flat_x(d, self.cols, self.relative)
        self.scaler = StandardScaler().fit(x, sample_weight=w)
        x = self.scaler.transform(x)
        cls = self.task == 'classification'
        y = d.drone_present.to_numpy(float) if cls else d.distance_m.to_numpy(float)
        torch.manual_seed(seed)
        net, dim = [], x.shape[1]
        for _ in range(self.depth):
            net += [nn.Linear(dim, self.width)]
            if self.dropout:
                net.append(nn.Dropout(self.dropout))
            net.append(nn.ReLU())
            dim = self.width
        net = nn.Sequential(*net, nn.Linear(dim, 1)).to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction='none') if cls else nn.SmoothL1Loss(reduction='none')
        tx = torch.as_tensor(x, dtype=torch.float32, device=device)
        ty = torch.as_tensor(y, dtype=torch.float32, device=device)
        tw = torch.as_tensor(w / w.mean(), dtype=torch.float32, device=device)
        if valid is not None:
            vx = torch.as_tensor(self.scaler.transform(_flat_x(valid, self.cols, self.relative)), dtype=torch.float32, device=device)
            vy = torch.as_tensor(valid.drone_present.to_numpy(float) if cls else valid.distance_m.to_numpy(float), device=device)
            vw = torch.as_tensor(weights(valid), dtype=torch.float32, device=device)
        best = np.inf
        for epoch in range(1, epochs + 1):
            net.train()
            for idx in torch.randperm(len(y), device=device).split(BATCH_MLP):
                optim.zero_grad()
                loss = (criterion(net(tx[idx]).flatten(), ty[idx]) * tw[idx]).mean()
                loss.backward()
                optim.step()
            net.eval()
            with torch.no_grad():
                tp = net(tx).flatten()
                row = {'epoch': epoch, 'train_loss': float((criterion(tp, ty) * tw).mean())}
                if cls:
                    row['train_accuracy'] = float((((tp >= 0) == ty.bool()).float() * tw).mean())
                else:
                    row['train_mae'] = float((abs(tp - ty) * tw).mean())
                if valid is None:
                    score = row['train_loss']
                else:
                    vp = net(vx).flatten()
                    score = float((criterion(vp, vy) * vw).sum() / vw.sum())
                    row['val_loss'] = score
                    if cls:
                        row['val_accuracy'] = float((((vp >= 0) == vy.bool()).float() * vw).sum() / vw.sum())
                    else:
                        row['val_mae'] = float((abs(vp - vy) * vw).sum() / vw.sum())
            self.curve.append(row)
            if not np.isfinite(score):
                raise ValueError('loss non finie')
            if valid is None or score < best - 1e-6:
                best, self.best_epoch = score, epoch
                self.layers = [(l.weight.detach().cpu().numpy().copy(), l.bias.detach().cpu().numpy().copy())
                               for l in net if isinstance(l, nn.Linear)]
            elif epoch - self.best_epoch >= patience:
                break
        return self

    def predict(self, d, *, bounded=True):
        x = self.scaler.transform(_flat_x(d, self.cols, self.relative))
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
        p = x.ravel()
        if self.task == 'classification':
            return 1 / (1 + np.exp(-np.clip(p, -60, 60)))
        return np.clip(p, 1., 5.) if bounded else p

    def parameter_count(self):
        return int(sum(w.size + b.size for w, b in self.layers))


class Linear:
    """Baseline : logistique (détection) ou ridge (distance)."""
    def __init__(self, task, kind=None, strength=1.0, relative=False):
        self.task = task
        self.kind = kind or ('logistic' if task == 'classification' else 'ridge')
        self.strength, self.relative = float(strength), bool(relative)
        self.curve, self.best_epoch = [], 0

    def fit(self, d, valid=None, device='cpu', epochs=0, patience=0, seed=0, cols=None):
        if cols is None:
            raise ValueError('préciser les colonnes de features')
        self.cols = list(cols)
        w = weights(d)
        x = _flat_x(d, self.cols, self.relative)
        self.scaler = StandardScaler().fit(x, sample_weight=w)
        x = self.scaler.transform(x)
        cls = self.task == 'classification'
        y = d.drone_present.to_numpy(int) if cls else d.distance_m.to_numpy(float)
        if cls and self.kind != 'logistic':
            raise ValueError('logistique uniquement en classification')
        if not cls and self.kind != 'ridge':
            raise ValueError('ridge uniquement en distance')
        if cls:
            self.estimator = LogisticRegression(C=self.strength, max_iter=1000)
        else:
            self.estimator = Ridge(alpha=self.strength)
        self.estimator.fit(x, y, sample_weight=w)
        return self

    def predict(self, d, *, bounded=True):
        x = self.scaler.transform(_flat_x(d, self.cols, self.relative))
        if self.task == 'classification':
            return self.estimator.predict_proba(x)[:, 1]
        p = self.estimator.predict(x)
        return np.clip(p, 1., 5.) if bounded else p

    def parameter_count(self):
        return int(np.size(self.estimator.coef_) + np.size(self.estimator.intercept_))


class Constant:
    """Référence distance : toujours 3 m (médiane de la plage 1-5 m)."""
    task = 'distance'
    curve, best_epoch = [], 0

    def fit(self, d, valid=None, **kwargs):
        return self

    def predict(self, d, *, bounded=True):
        return np.full(len(d), 3.0)

    def parameter_count(self):
        return 0


MODELS = {
    ('classification', 'cnn'): lambda a: CnnPresence(**a),
    ('classification', 'mlp'): lambda a: Mlp('classification', **a),
    ('classification', 'logistic'): lambda a: Linear('classification', kind='logistic', **a),
    ('distance', 'mlp'): lambda a: Mlp('distance', **a),
    ('distance', 'ridge'): lambda a: Linear('distance', kind='ridge', **a),
    ('distance', 'constant'): lambda a: Constant(),
}


def build(task, name, **params):
    if (task, name) not in MODELS:
        raise ValueError(f"modèle {name!r} inconnu pour la tâche {task!r} : {sorted(MODELS)}")
    return MODELS[(task, name)](params)
