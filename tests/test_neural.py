"""Synthetic neural tests: no real capture, no holdout take, no CUDA required."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from edth.models import AUDIO, LOGMEL_SHAPE, Model, weights

ROOT = Path(__file__).resolve().parents[1]


def cnn_params(**overrides):
    params = dict(family='cnn', width=8, depth=1, lr=.01, dropout=0.0,
                  relative=False, bands='full', phone=False, batch_size=64)
    params.update(overrides)
    return params


def corpus(seed=7, groups=9):
    """Synthetic windows; each take group carries negatives and 1/3/5 m positives."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        for take, distance in enumerate([None, 1., 3., 5.]):
            present = distance is not None
            for _ in range(1 + take):
                pattern = rng.normal(0., 1., LOGMEL_SHAPE)
                level = -30. - 6. * distance if present else -55.
                if present:
                    pattern[22:28] += 2.0  # synthetic high-band drone signature
                rows.append(dict(
                    sample_id=f'{group}-{take}', group_id=f'g{group}',
                    device_id=f'D{group % 2}', fold=group % 3, split='dev',
                    drone_present=present, distance_m=distance,
                    logmel=np.ravel(-80. + 4. * pattern).astype(float),
                    **{c: level + i * .1 for i, c in enumerate(AUDIO)}))
    return pd.DataFrame(rows)


def split(frame):
    return frame[frame.fold != 0], frame[frame.fold == 0]


def test_cnn_refit_without_validation_runs_exact_epochs_and_uses_global_mean_weight():
    import torch
    from torch import nn
    frame = corpus(); train, _ = split(frame)
    model = Model('classification', cnn_params()).fit(train, valid=None, epochs=5, patience=1, seed=9)
    # No validation fold: patience must not stop the final refit before the requested epoch.
    assert len(model.curve) == 5
    assert model.best_epoch == 5

    # Loss convention must match the MLP: weights normalized by the global train mean.
    net = model._cnn_net()
    net.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in model.cnn_state.items()})
    net.eval()
    x = torch.from_numpy(model._scale_cnn(model._cnn_input(train))).unsqueeze(1)
    y = torch.from_numpy(train.drone_present.to_numpy(np.float32))
    w = weights(train); w = (w / w.mean()).astype(np.float32)
    with torch.inference_mode():
        logits = net(x).flatten()
    expected = float((nn.BCEWithLogitsLoss(reduction='none')(logits, y) * torch.from_numpy(w)).mean())
    assert model.curve[-1]['train_loss'] == pytest.approx(expected, abs=1e-5)


def test_cnn_fit_predict_interfaces_and_parameter_count():
    frame = corpus(); train, valid = split(frame)
    model = Model('classification', cnn_params()).fit(train, valid, 'cpu', epochs=3, patience=2, seed=1)

    assert model.devices == sorted(frame.device_id.unique())
    assert model.best_epoch >= 1
    assert len(model.curve) >= 1
    for row in model.curve:
        assert {'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'} <= set(row)
    probabilities = model.predict(valid)
    assert probabilities.shape == (len(valid),)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()

    # The stored NumPy state must rebuild the exact same module for a CPU reload.
    assert set(model.cnn_state) == set(model._cnn_net().state_dict())
    assert all(isinstance(v, np.ndarray) for v in model.cnn_state.values())
    assert model.parameter_count() == sum(v.size for v in model.cnn_state.values())
    assert model.parameter_count() > 0

    import torch
    assert not any(isinstance(m, torch.nn.Dropout2d) for m in model._cnn_net().modules())
    dropped = Model('classification', cnn_params(dropout=.3))
    assert any(isinstance(m, torch.nn.Dropout2d) for m in dropped._cnn_net().modules())


def test_cnn_band_scaler_is_train_only_and_take_weighted():
    frame = corpus(); train, valid = split(frame)
    model = Model('classification', cnn_params()).fit(train, valid, 'cpu', epochs=1, patience=1, seed=2)

    x = np.stack([np.asarray(v, float).reshape(LOGMEL_SHAPE) for v in train.logmel])
    w = weights(train)
    mean = (w[:, None] * x.mean(axis=2)).sum(0) / w.sum()
    np.testing.assert_allclose(model.lm_mean, mean, rtol=0, atol=1e-10)
    assert model.lm_std.shape == (LOGMEL_SHAPE[0],)
    assert np.isfinite(model.lm_std).all() and (model.lm_std > 0).all()

    # Mutating validation audio must not move the train-only standardisation.
    poisoned = valid.copy()
    poisoned['logmel'] = [np.full(v.size, 500.) for v in poisoned.logmel]
    other = Model('classification', cnn_params()).fit(train, poisoned, 'cpu', epochs=1, patience=1, seed=2)
    np.testing.assert_array_equal(model.lm_mean, other.lm_mean)
    np.testing.assert_array_equal(model.lm_std, other.lm_std)


def test_cnn_relative_removes_the_broadband_level():
    frame = corpus(); train, valid = split(frame)
    model = Model('classification', cnn_params(relative=True)).fit(train, valid, 'cpu', epochs=2, patience=1, seed=3)
    shifted = valid.copy()
    shifted['logmel'] = [np.asarray(v) - 12. for v in shifted.logmel]
    shifted['rms_db'] = shifted.rms_db - 12.
    np.testing.assert_allclose(model.predict(valid), model.predict(shifted), rtol=0, atol=1e-6)

    absolute = Model('classification', cnn_params(relative=False)).fit(train, valid, 'cpu', epochs=2, patience=1, seed=3)
    assert not np.allclose(absolute.predict(valid), absolute.predict(shifted), atol=1e-6)


def test_cnn_eval_is_deterministic_and_batch_invariant():
    frame = corpus(); train, valid = split(frame)
    model = Model('classification', cnn_params(dropout=.3)).fit(train, valid, 'cpu', epochs=2, patience=1, seed=4)
    reference = model.predict(valid)
    np.testing.assert_array_equal(model.predict(valid), reference)  # dropout is off at eval
    model.p['batch_size'] = 7
    np.testing.assert_allclose(model.predict(valid), reference, rtol=0, atol=1e-6)


def test_cnn_joblib_roundtrip_in_subprocess(tmp_path):
    frame = corpus(); train, valid = split(frame)
    model = Model('classification', cnn_params()).fit(train, valid, 'cpu', epochs=2, patience=1, seed=5)
    expected = model.predict(valid)
    joblib.dump(model, tmp_path / 'model.joblib')
    valid.to_parquet(tmp_path / 'windows.parquet')

    script = tmp_path / 'reload.py'
    script.write_text(
        'import sys\n'
        'import joblib, numpy as np, pandas as pd, torch\n'
        'def _no_cuda(*a, **k): raise AssertionError("prediction must not use CUDA")\n'
        'torch.nn.Module.cuda = _no_cuda\n'
        'try:\n'
        '    torch.Tensor.cuda = _no_cuda\n'
        'except (AttributeError, TypeError):\n'
        '    pass\n'
        'm = joblib.load(sys.argv[1]); d = pd.read_parquet(sys.argv[2])\n'
        'assert m.p["family"] == "cnn"\n'
        'assert m.cnn_state and all(isinstance(v, np.ndarray) for v in m.cnn_state.values())\n'
        'np.save(sys.argv[3], m.predict(d))\n'
    )
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run(
        [sys.executable, str(script), str(tmp_path / 'model.joblib'),
         str(tmp_path / 'windows.parquet'), str(tmp_path / 'pred.npy')],
        check=True, cwd=ROOT, env=env, timeout=300)
    np.testing.assert_allclose(np.load(tmp_path / 'pred.npy'), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize('task', ['classification', 'distance'])
def test_mlp_dropout_on_both_tasks(task):
    frame = corpus()
    if task == 'distance':
        frame = frame[frame.drone_present]
    train, valid = split(frame)
    params = dict(family='mlp', width=8, depth=2, lr=.01, dropout=.3, bands='full', phone=False)
    if task == 'classification':
        params['relative'] = True
    model = Model(task, params).fit(train, valid, 'cpu', epochs=3, patience=1, seed=6)
    predictions = model.predict(valid)
    assert predictions.shape == (len(valid),)
    assert np.isfinite(predictions).all()
    lower, upper = (0., 1.) if task == 'classification' else (1., 5.)
    assert ((predictions >= lower) & (predictions <= upper)).all()
    np.testing.assert_array_equal(predictions, model.predict(valid))
    assert model.parameter_count() > 0

    # Older parameter dictionaries without the key keep the no-dropout behaviour.
    Model(task, {k: v for k, v in params.items() if k != 'dropout'}).fit(
        train, valid, 'cpu', epochs=1, patience=1, seed=6)


def test_parameter_count_for_every_family():
    frame = corpus(); train = frame[frame.fold != 0]; positive = train[train.drone_present]
    assert Model('distance', {'family': 'constant'}).fit(train).parameter_count() == 0
    ridge = Model('distance', {'family': 'ridge', 'bands': 'full'}).fit(positive)
    assert ridge.parameter_count() == len(AUDIO) + 1
    calibration = Model('distance', {'family': 'calibration', 'bands': '4-8k'}).fit(positive)
    assert calibration.parameter_count() == 2
    logistic = Model('classification', {'family': 'logistic', 'bands': 'full', 'relative': True}).fit(train)
    assert logistic.parameter_count() == len(AUDIO) - 1 + 1
    mlp = Model('classification', {'family': 'mlp', 'width': 8, 'depth': 2, 'lr': .01,
                                  'bands': 'full', 'relative': True}).fit(train)
    assert mlp.parameter_count() == sum(w.size + b.size for w, b in mlp.layers)
    cnn = Model('classification', cnn_params()).fit(train, epochs=1, patience=1, seed=0)
    assert cnn.parameter_count() == sum(v.size for v in cnn.cnn_state.values())


def test_cnn_rejects_unsupported_configurations():
    frame = corpus(); train = frame[frame.fold != 0]
    with pytest.raises(ValueError, match='classification'):
        Model('distance', cnn_params()).fit(train, epochs=1)
    with pytest.raises(ValueError, match='log-mel'):
        Model('classification', cnn_params(bands='4-8k')).fit(train, epochs=1)
    with pytest.raises(ValueError, match='phone'):
        Model('classification', cnn_params(phone=True)).fit(train, epochs=1)
    with pytest.raises(ValueError, match='width'):
        Model('classification', cnn_params(width=64)).fit(train, epochs=1)
    with pytest.raises(ValueError, match='depth'):
        Model('classification', cnn_params(depth=3)).fit(train, epochs=1)
    with pytest.raises(ValueError, match='dropout'):
        Model('classification', cnn_params(dropout=.8)).fit(train, epochs=1)
    with pytest.raises(ValueError, match='logmel'):
        Model('classification', cnn_params()).fit(train.drop(columns='logmel'), epochs=1)
    broken = train.copy()
    broken['logmel'] = [np.zeros(700) for _ in broken.logmel]
    with pytest.raises(ValueError, match='704'):
        Model('classification', cnn_params()).fit(broken, epochs=1)


def test_cross_validate_accepts_the_cnn_interface():
    from edth.train import cross_validate
    args = SimpleNamespace(device='cpu', epochs=2, patience=1, seed=11)
    result = cross_validate(corpus(), 'classification', cnn_params(), args)
    assert np.isfinite(result['value'])
    assert 0 <= result['value'] <= 1
    assert set(result['curves']) == {'0', '1', '2'}
    assert all(len(row) >= 1 for row in result['curves'].values())
    assert result['params']['family'] == 'cnn'
