import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from edth.train import prepare, run
from edth.models import AUDIO, Model, weights


def corpus():
    rows=[]
    for g in range(9):
        for take,dist in enumerate([None,1.,3.,5.]):
            for w in range(1+take):
                level=-60. if dist is None else -15.-dist
                rows.append(dict(sample_id=f'{g}-{take}',group_id=str(g),device_id=f'D{g%2}',fold=g%3,
                            split='dev' if g<6 else 'test',drone_present=dist is not None,distance_m=dist,
                            logmel=(level+np.linspace(0,1,704,dtype='float32')+w*.01),
                            **{c:level+i*.1+w*.01 for i,c in enumerate(AUDIO)}))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('task',['distance','classification'])
def test_training_smoke(tmp_path,task,monkeypatch):
    original_fit = Model.fit
    def guarded_fit(self, train, valid=None, *args, **kwargs):
        assert set(train.split) == {'dev'}
        if valid is not None:
            assert set(valid.split) == {'dev'}
            assert set(valid.group_id).isdisjoint(train.group_id)
        return original_fit(self, train, valid, *args, **kwargs)
    monkeypatch.setattr(Model, 'fit', guarded_fit)
    d=corpus(); data=tmp_path/'windows.parquet'; out=tmp_path/task; d.to_parquet(data)
    args=SimpleNamespace(data=data,out=out,device='cpu',task=task,trials=1,epochs=2,patience=1,seed=4)
    run(args)
    m=json.loads((out/'manifest.json').read_text()); assert m['windows_sha256']==hashlib.sha256(data.read_bytes()).hexdigest()
    results=json.loads((out/'metrics.json').read_text()); assert set(results['test']['by_device'])=={'D0','D1'}
    curves=json.loads((out/'curves.json').read_text()); assert any(x['params']['family']==('cnn' if task=='classification' else 'mlp') and len(x['folds']['0']) for x in curves['cv'])
    cv = pd.read_csv(out/'cv_comparison.csv')
    assert cv.selected.sum() == 1 and {'fold_0','fold_1','fold_2','dropout'} <= set(cv)
    predictions=pd.read_parquet(out/'test_predictions.parquet'); assert set(predictions.group_id)=={'6','7','8'}
    model=joblib.load(out/'model.joblib'); _,test=prepare(d,task)
    np.testing.assert_allclose(predictions.pred,model.predict(test[m['feature_columns']+['device_id']]))
    with pytest.raises(FileExistsError): run(args)


def test_dropout_ablation_and_old_cache_error():
    from edth.train import initial_trials, suggest
    from optuna.trial import FixedTrial
    for task, families in [('classification', {'cnn','mlp'}), ('distance', {'mlp'})]:
        params=[suggest(FixedTrial(p),task) for p in initial_trials(task)]
        assert {p['family'] for p in params} == families
        for family in families:
            assert {p['dropout'] for p in params if p['family']==family} == {0.,.15,.30}
    with pytest.raises(ValueError, match='rebuild'):
        prepare(corpus().drop(columns='logmel'), 'classification')


def test_refuses_group_leakage():
    d=corpus(); d.loc[d.index[0],'split']='test'
    with pytest.raises(ValueError,match='group'): prepare(d,'distance')


def test_training_only_references_and_weights():
    d=corpus(); dev,test=prepare(d,'distance'); test=test.copy(); test.b4_8=100.
    np.testing.assert_allclose(dev.assign(w=weights(dev)).groupby('sample_id').w.sum(),1.)
    model=Model('distance',{'family':'calibration','bands':'4-8k'}).fit(dev,test)
    assert all(x<0 for x in model.refs.values())
    extra=d[d.drone_present].copy(); extra['distance_m']=.1; extra['sample_id']+='close'
    dev,_=prepare(pd.concat([d,extra]),'distance'); assert dev.distance_m.min()==1.
