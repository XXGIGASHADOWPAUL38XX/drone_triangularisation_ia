"""Plot the strongest completed neural candidate on development folds only."""
import argparse
import json
from pathlib import Path


def plot_run(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=json.loads((run/'curves.json').read_text())['cv']
    dev=json.loads((run/'metrics.json').read_text())['development']
    cls=json.loads((run/'manifest.json').read_text())['task']=='classification'
    neural=[(c,r) for c,r in zip(curves,dev,strict=True) if c['params']['family'] in ('mlp','cnn')]
    if not neural:
        print('No completed neural trial: no epoch curves to plot.'); return
    candidate,_=(max if cls else min)(neural,key=lambda pair:pair[1]['value'])
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    metric='val_accuracy' if cls else 'val_mae'
    for fold,rows in candidate['folds'].items():
        epochs=[r['epoch'] for r in rows]
        line,=axes[0].plot(epochs,[r['train_loss'] for r in rows],label=f'Train {fold}')
        axes[0].plot(epochs,[r['val_loss'] for r in rows],'--',color=line.get_color(),label=f'Val {fold}')
        axes[1].plot(epochs,[r[metric] for r in rows],label=f'Val {fold}')
    for ax,title in zip(axes,['Loss',metric]):
        ax.set(xlabel='Époque',title=title); ax.legend(fontsize=8)
    fig.suptitle(f"Meilleur réseau ({candidate['params']['family'].upper()}) en validation croisée — pas le score de test"); fig.savefig(run/'learning-curves.png'); plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',required=True,type=Path); plot_run(ap.parse_args().run)
