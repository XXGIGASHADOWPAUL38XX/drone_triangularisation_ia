"""End-to-end diagnostics on frozen holdout takes, never used as an HPO objective."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .release import digest, verify_bundle


def evaluate(bundle, parquet):
    info = verify_bundle(bundle)
    if digest(parquet) != info['windows_sha256']:
        raise ValueError('Expected the release evaluation corpus')
    features = pd.read_parquet(parquet, filters=[('split', '=', 'test')]).sort_values(['sample_id','start_sample_24k'])
    clf, reg = (joblib.load(bundle/t/'model.joblib') for t in ('classification','distance'))
    features['p'] = clf.predict(features); features['triggered'] = features.p >= .5
    eligible = features.drone_present & features.distance_m.between(1,5)
    accepted = eligible & features.triggered
    features['error'] = np.nan
    if accepted.any():
        features.loc[accepted,'error'] = abs(reg.predict(features.loc[accepted]) - features.loc[accepted,'distance_m'])
    takes = json.loads((bundle/'manifest.json').read_text())['takes']
    durations = {r['sample_id']: r['duration_s'] for r in takes if r['split']=='test'}
    rows = []
    for sid, part in features.groupby('sample_id'):
        first = part.iloc[0]; starts = part.start_sample_24k.to_numpy(); detected = part.triggered.to_numpy()
        contiguous = np.r_[False, np.diff(starts) == 6000]
        rising = detected & ~(contiguous & np.r_[False,detected[:-1]])
        # Sum union of 1s valid windows, not the sum of overlapping durations.
        valid_s = 1.0 + np.minimum(1.0,np.diff(starts)/24000).sum()
        rows.append(dict(device_id=first.device_id,positive=bool(first.drone_present),eligible=bool(1<=first.distance_m<=5),
                         observed_s=durations[sid], valid_s=valid_s, windows=len(part), detections=int(detected.sum()),
                         alerts=int(rising.sum()), mae=part.error.mean()))
    for r in takes:
        if r['split']=='test' and not r['valid_windows'] and r.get('reason')=='no valid window':
            rows.append(dict(device_id=r['device_id'],positive=r['drone_present'],eligible=bool(r['distance_m'] is not None and 1<=r['distance_m']<=5),observed_s=r['duration_s'],valid_s=0,windows=0,detections=0,alerts=0,mae=np.nan))
    def summary(rs):
        neg = [r for r in rs if not r['positive']]; pos = [r for r in rs if r['eligible']]
        def ratio(a,b): return float(a/b) if b else None
        errs=[r['mae'] for r in pos if np.isfinite(r['mae'])]
        return dict(takes=len(rs),valid_audio_fraction=ratio(sum(r['valid_s'] for r in rs),sum(r['observed_s'] for r in rs)),
                    false_alert_starts_per_valid_negative_minute=ratio(60*sum(r['alerts'] for r in neg),sum(r['valid_s'] for r in neg)),
                    range_coverage_on_valid_positive_windows=ratio(sum(r['detections'] for r in pos),sum(r['windows'] for r in pos)),
                    takes_with_distance=len(errs),eligible_takes=len(pos),distance_mae_m_equal_take_weight=float(np.mean(errs)) if errs else None)
    return {'scope':'Same-campaign grouped holdout. Fixed 0.5 threshold, no smoothing. Take labels, no event-onset annotations.',
            'overall':summary(rows),'by_device':{d:summary([r for r in rows if r['device_id']==d]) for d in sorted({r['device_id'] for r in rows})}}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--bundle',required=True,type=Path); ap.add_argument('--data',required=True,type=Path); a=ap.parse_args()
    print(json.dumps(evaluate(a.bundle,a.data),indent=2,allow_nan=False))
