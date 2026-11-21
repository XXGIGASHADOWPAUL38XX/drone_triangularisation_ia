"""Exercise both full pipelines using only development takes, keeping the real test sealed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); data=a.out/'input'; data.mkdir()
    df=pd.read_parquet(a.data/'windows.parquet',filters=[('split','=','dev')]); test_fold=min(df.fold.unique()); df.loc[df.fold==test_fold,'split']='test'; df.loc[df.split=='test','fold']=-1
    df.to_parquet(data/'windows.parquet',index=False)
    manifest=json.loads((a.data/'manifest.json').read_text()); ids=df.drop_duplicates('sample_id').set_index('sample_id')[['split','fold']].to_dict('index')
    manifest['takes']=[{**r,**ids[r['sample_id']]} for r in manifest['takes'] if r['sample_id'] in ids]; manifest['purpose']='CPU plumbing test only; original test excluded'; (data/'manifest.json').write_text(json.dumps(manifest))
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':str((a.out/'mpl').resolve())}
    def run(*args): subprocess.run([sys.executable,'-m',*map(str,args)],check=True,env=env)
    for task in ('classification','distance'):
        run('edth.train','--task',task,'--data',data/'windows.parquet','--out',a.out/task,'--trials',2 if task=='classification' else 1,'--epochs',2,'--device','cpu')
        run('edth.plot','--run',a.out/task)
    run('edth.release','pack','--data',data,'--classification',a.out/'classification','--distance',a.out/'distance','--out',a.out/'bundle')
    run('edth.evaluate','--data',data/'windows.parquet','--bundle',a.out/'bundle')


if __name__=='__main__': main()
