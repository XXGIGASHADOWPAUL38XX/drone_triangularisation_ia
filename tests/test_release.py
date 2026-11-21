import hashlib
from pathlib import Path
import json
import wave

import numpy as np
import pandas as pd
import pytest

from edth.release import build, digest, verify_bundle


def test_pack_hashes_and_refuses_mixed_datasets(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    f = data/'windows.parquet'; pd.DataFrame({'device_id':['test']}).to_parquet(f)
    for task in ('classification','distance'):
        folder = tmp_path/task; folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps({'task':task,'windows_sha256':digest(f),'device_ids':['test'],'training_code_sha256':hashlib.sha256(Path('edth/train.py').read_bytes()+Path('edth/models.py').read_bytes()).hexdigest()}))
        for name in ('metrics.json','curves.json','model.joblib'):
            (folder/name).write_text('{}')
    c,d = tmp_path/'classification',tmp_path/'distance'
    build(data,c,d,tmp_path/'bundle')
    assert verify_bundle(tmp_path/'bundle')['device_ids']==['test']
    (tmp_path/'bundle/classification/model.joblib').write_text('bad')
    with pytest.raises(ValueError,match='mismatch'): verify_bundle(tmp_path/'bundle')
    with pytest.raises(ValueError,match='Wrong task'): build(data,d,c,tmp_path/'mismatch')


def test_replay_uses_gate_and_quality_mask(tmp_path):
    from edth.predict import predict_file
    class Constant:
        def __init__(self, v): self.v=v
        def predict(self, df): return np.full(len(df),self.v)
    path=tmp_path/'master.wav'
    def write(x):
        with wave.open(str(path),'wb') as f:
            f.setparams((1,2,48000,0,'NONE','not compressed')); f.writeframes(x.astype('<i2').tobytes())
    write(np.tile([130,-130],72000))
    rows=predict_file(path,'dev',Constant(.2),Constant(4))
    assert rows and all(r['distance_m'] is None for r in rows)
    assert all(r['distance_m']==4 for r in predict_file(path,'dev',Constant(.8),Constant(4)))
    write(np.zeros(144000)); assert predict_file(path,'dev',Constant(.8),Constant(4))==[]
