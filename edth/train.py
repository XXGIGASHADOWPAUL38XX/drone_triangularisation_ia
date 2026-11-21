"""Grouped, take-weighted model selection; python -m edth.train --help."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, mean_absolute_error
from .models import AUDIO, Model, weights


def prepare(d, task):
    required = {'sample_id','device_id','group_id','split','fold','drone_present','distance_m',*AUDIO}
    if task == 'classification' and 'logmel' not in d:
        raise ValueError('Missing logmel spectrogram: rebuild the feature cache with edth.data in a new directory')
    missing = required-set(d)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    keys = ['sample_id','device_id','group_id','split','drone_present']
    if d[keys].isna().any().any() or not d.split.isin(['dev','test']).all(): raise ValueError('Invalid identifiers/split/label')
    if not pd.api.types.is_bool_dtype(d.drone_present): raise ValueError('drone_present must be boolean')
    if (d.groupby('sample_id')[['group_id','device_id','drone_present','distance_m']].nunique(dropna=False)>1).any().any():
        raise ValueError('Inconsistent take metadata')
    if (d.groupby('group_id').split.nunique()>1).any(): raise ValueError('group crosses dev/test')
    if not np.isfinite(d[AUDIO].to_numpy(float)).all(): raise ValueError('Non-finite audio features')
    if d.loc[d.drone_present,'distance_m'].isna().any() or d.loc[~d.drone_present,'distance_m'].notna().any(): raise ValueError('Inconsistent presence/distance')
    dev = d[d.split=='dev']; folds = pd.to_numeric(dev.fold, errors='coerce')
    if not np.isfinite(folds).all() or (folds%1!=0).any() or (dev.groupby('group_id').fold.nunique()>1).any(): raise ValueError('Invalid grouped folds')
    if task == 'distance': d = d[d.drone_present & d.distance_m.between(1,5)]
    dev, test = d[d.split=='dev'].copy(), d[d.split=='test'].copy()
    dev['fold'] = dev.fold.astype(int)
    if dev.fold.nunique()<2 or test.empty: raise ValueError('Need at least two development folds and reserved test takes')
    for fold in dev.fold.unique():
        train = dev[dev.fold!=fold]
        if task=='classification' and train.drone_present.nunique()!=2: raise ValueError(f'Training fold excluding {fold} lacks a class')
    return dev,test


def metrics(d,p,task):
    w=weights(d); result={'takes':int(d.sample_id.nunique()),'windows':len(d)}
    if task=='distance': return dict(result, mae_m=float(mean_absolute_error(d.distance_m,p,sample_weight=w)),bias_m=float(np.average(p-d.distance_m,weights=w)))
    y=d.drone_present; pred=p>=.5
    return dict(result, accuracy=float(accuracy_score(y,pred,sample_weight=w)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred,sample_weight=w)) if y.nunique()==2 else None,
                precision=float(precision_score(y,pred,sample_weight=w,zero_division=0)),recall=float(recall_score(y,pred,sample_weight=w,zero_division=0)))


def report(d,task):
    result = {'overall': metrics(d,d.pred.to_numpy(),task),
              'by_device':{str(k):metrics(v,v.pred.to_numpy(),task) for k,v in d.groupby('device_id')}}
    if task=='distance': result['by_distance'] = {str(k):metrics(v,v.pred.to_numpy(),task) for k,v in d.groupby('distance_m')}
    return result


def cross_validate(dev,task,params,a,trial=None):
    outputs=[]; curves={}; epochs=[]; fold_scores={}
    metric_name = 'mae_m' if task == 'distance' else 'balanced_accuracy'
    for step,f in enumerate(sorted(dev.fold.unique())):
        tr,va=dev[dev.fold!=f],dev[dev.fold==f]
        model=Model(task,params).fit(tr,va,a.device,a.epochs,a.patience,a.seed+int(f))
        predictions = model.predict(va)
        outputs.append(va.assign(pred=predictions)); curves[str(f)]=model.curve
        fold_scores[f'fold_{f}'] = metrics(va, predictions, task)[metric_name]
        if params['family'] in ('mlp', 'cnn'): epochs.append(model.best_epoch)
        if trial is not None:
            tmp=pd.concat(outputs); m=metrics(tmp,tmp.pred.to_numpy(),task)
            value=m['mae_m' if task=='distance' else 'balanced_accuracy']
            if value is not None:
                trial.report(value,step)
                if step+1 < dev.fold.nunique() and trial.should_prune(): raise optuna.TrialPruned()
    stats=report(pd.concat(outputs),task)
    value=stats['overall']['mae_m' if task=='distance' else 'balanced_accuracy']
    return {'params':params,'value':value,'cv':stats,'fold_scores':fold_scores,'curves':curves,'epochs':max(1,int(np.median(epochs))) if epochs else 1}


def suggest(t,task):
    cls = task == 'classification'
    family = t.suggest_categorical('family', ['cnn','mlp','logistic'] if cls else ['mlp','ridge'])
    p = {'family': family}
    if cls:
        p['relative'] = t.suggest_categorical('relative', [False, True])
    if family == 'cnn':
        # Full time-frequency input, no phone metadata: test the audio hypothesis directly.
        p.update(bands='full', phone=False, width=t.suggest_categorical('cnn_channels', [8,16,32]),
                 depth=t.suggest_int('cnn_blocks', 1, 2))
    else:
        bands = ['full','low_mel','mid_mel','4-8k'] + ([] if cls else ['rms'])
        p.update(bands=t.suggest_categorical('bands', bands), phone=t.suggest_categorical('phone', [False,True]))
        if family == 'mlp':
            p.update(width=t.suggest_categorical('width',[8,16,32,64]), depth=t.suggest_int('depth',1,3))
    if family in ('cnn', 'mlp'):
        p.update(lr=t.suggest_float('lr',.0003,.02,log=True), dropout=t.suggest_categorical('dropout',[0.,.15,.30]))
    else:
        p['strength'] = t.suggest_float('strength',.01,100,log=True)
    return p


def initial_trials(task):
    """Matched dropout ablations before adaptive search, within the supplied budget."""
    cls = task == 'classification'
    for rate in (0., .15, .30):
        if cls:
            yield dict(family='cnn', cnn_channels=8, cnn_blocks=2, lr=.003, dropout=rate, relative=True)
        yield dict(family='mlp', width=16, depth=1, lr=.003, dropout=rate, bands='full', phone=False,
                   **({'relative':True} if cls else {}))


def dump(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def run(a):
    if min(a.epochs,a.patience)<1 or a.trials<0: raise ValueError('Invalid training budget')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise FileExistsError('Already finalized. Do not re-tune on a consulted holdout.')
    if a.device=='cuda':
        import torch
        if not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    dev,test=prepare(pd.read_parquet(a.data),a.task)
    sha=hashlib.sha256()
    with open(a.data,'rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''): sha.update(block)
    source_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('models.py').read_bytes()).hexdigest()
    sig={'version':1,'training_code_sha256':source_hash,'windows_sha256':sha.hexdigest(),'task':a.task,'seed':a.seed,'epochs':a.epochs,'patience':a.patience,'device':a.device}
    cls=a.task=='classification'; metric='balanced_accuracy' if cls else 'mae_m'; key=(max if cls else min)
    fixed=[{'family':'logistic','bands':'full','relative':True}] if cls else [{'family':'constant'},{'family':'calibration','bands':'4-8k'},{'family':'calibration','bands':'4-8k','phone':True},{'family':'ridge','bands':'full'}]
    study=optuna.create_study(study_name=a.task,storage='sqlite:///'+str(out/'study.sqlite3'),load_if_exists=True,
                direction='maximize' if cls else 'minimize', sampler=optuna.samplers.TPESampler(seed=a.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=6,n_warmup_steps=1))
    if study.user_attrs.get('signature',sig)!=sig: raise ValueError('Cannot resume with different data or training configuration')
    study.set_user_attr('signature',sig)
    candidates=[cross_validate(dev,a.task,p,a) for p in fixed]
    def objective(trial):
        r=cross_validate(dev,a.task,suggest(trial,a.task),a,trial); trial.set_user_attr('result',r); return r['value']
    if not study.trials:
        for params in list(initial_trials(a.task))[:a.trials]:
            study.enqueue_trial(params)
    done=sum(t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.PRUNED] for t in study.trials)
    if a.trials>done: study.optimize(objective,n_trials=a.trials-done)
    candidates.extend(t.user_attrs['result'] for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE)
    winner=key(candidates,key=lambda r:r['value']) # ties keep fixed baseline first
    # Compare architectures/dropout on development, before evaluating the reserved takes.
    pd.DataFrame([{**r['params'], metric:r['value'], **r['fold_scores'], 'selected':r is winner}
                  for r in candidates]).to_csv(out/'cv_comparison.csv', index=False)
    model=Model(a.task,winner['params']).fit(dev,device=a.device,epochs=winner['epochs'],seed=a.seed)
    pred=test.assign(pred=model.predict(test)); joblib.dump(model,out/'model.joblib')
    test_summary = report(pred,a.task)
    # Diagnostic reference scores only; these never feed selection or refitting.
    test_baselines = []
    for params in fixed:
        ref = Model(a.task, params).fit(dev)
        test_baselines.append({'params': params, 'metrics': report(test.assign(pred=ref.predict(test)), a.task)})
    if not cls:
        pred['raw_pred'] = model.predict(test,bounded=False)
        test_summary['unbounded'] = report(pred.assign(pred=pred.raw_pred),a.task)
    columns = ['sample_id','group_id','device_id','drone_present','distance_m','pred'] + ([] if cls else ['raw_pred'])
    if 'start_sample_24k' in pred: columns.append('start_sample_24k')
    pred[columns].to_parquet(out/'test_predictions.parquet',index=False)
    dump(out/'curves.json',{'cv':[{'params':r['params'],'folds':r['curves']} for r in candidates],'final_train':model.curve})
    dump(out/'metrics.json',{'weighting':'equal total weight per take; window predictions','development':[{k:v for k,v in r.items() if k!='curves'} for r in candidates],'test':test_summary,'test_baselines':test_baselines})
    family = winner['params']['family']
    parameter_count = model.parameter_count()
    feature_columns = ['logmel', 'rms_db'] if family == 'cnn' else AUDIO
    dump(out/'manifest.json',{**sig,'estimator_parameters':int(parameter_count),'params':winner['params'],'selection_metric':metric,'cv_score':winner['value'],'epochs_final':winner['epochs'],
            'model_file':'model.joblib','feature_columns':feature_columns,'device_ids':model.devices,'threshold':.5 if cls else None,
            'output':'p_present' if cls else 'distance_m_conditional_on_presence','distance_domain_m':None if cls else [1,5],
            'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','optuna','joblib','pyarrow']+(['torch'] if family in ('mlp','cnn') else [])},'python':platform.python_version()})
    return winner['params']


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--task',choices=['classification','distance'],required=True); p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    for name,default in [('trials',20),('epochs',60),('patience',8),('seed',42)]: p.add_argument('--'+name,type=int,default=default)
    a=p.parse_args(); print(run(a))


if __name__=='__main__': main()
