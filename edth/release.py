"""Create a self-contained release; publishing is separate and explicitly opt-in."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile


def digest(path):
    with path.open('rb') as fp:
        return hashlib.file_digest(fp, 'sha256').hexdigest()


def verify_bundle(root):
    manifest = json.loads((root/'release.json').read_text())
    for rel, expected in manifest['sha256'].items():
        if digest(root/rel) != expected:
            raise ValueError(f'Artifact mismatch: {rel}')
    # Refuse to run stale preprocessing / model code from a different checkout.
    for name in ('data.py', 'models.py'):
        rel = 'edth/' + name
        if rel in manifest['sha256'] and digest(Path(__file__).with_name(name)) != manifest['sha256'][rel]:
            raise ValueError(f'Use the code bundled in this release: {rel}')
    return manifest


def build(data, classification, distance, out):
    if out.exists():
        raise FileExistsError(f'Choose a fresh release directory: {out}')
    parquet = data/'windows.parquet'
    parquet_hash = digest(parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        root = Path(tmp)/'bundle'; root.mkdir()
        pkg = Path(__file__).parent
        training_hash = hashlib.sha256((pkg/'train.py').read_bytes() + (pkg/'models.py').read_bytes()).hexdigest()
        trained_devices = []
        for task, source in [('classification', classification), ('distance', distance)]:
            run = json.loads((source/'manifest.json').read_text())
            if run.get('task') != task or run.get('windows_sha256') != parquet_hash:
                raise ValueError(f'Wrong task or dataset for {source}')
            if run.get('training_code_sha256') != training_hash:
                raise ValueError('Training and bundled model code differ')
            trained_devices.append(set(run['device_ids']))
            folder = root/task; folder.mkdir()
            for name in ('model.joblib','manifest.json','metrics.json','curves.json'):
                shutil.copyfile(source/name, folder/name)
            if (source/'cv_comparison.csv').is_file():
                shutil.copyfile(source/'cv_comparison.csv', folder/'cv_comparison.csv')
        pkg = Path(__file__).parent
        (root/'edth').mkdir()
        for path in sorted(pkg.glob('*.py')):
            shutil.copyfile(path, root/'edth'/path.name)
        for path in (pkg.parent/'README.md',pkg.parent/'requirements.txt',pkg.parent/'requirements-cpu.lock'):
            shutil.copyfile(path, root/path.name)
        for name in ('manifest.json','splits.csv','preprocessing.json'):
            if (data/name).is_file():
                shutil.copyfile(data/name, root/name)
        device_ids = sorted(set.intersection(*trained_devices))
        versions = {p:metadata.version(p) for p in ('numpy','scipy','pyarrow','pandas','scikit-learn','joblib','torch','optuna','huggingface-hub','python-dotenv')}
        (root/'environment.txt').write_text(''.join(f'{p}=={v}\n' for p,v in versions.items()))
        info = dict(format=1, sample_rate=24000, device_ids=device_ids, python=platform.python_version(), windows_sha256=parquet_hash,
                    scope='Controlled demo, known devices in hand, 1–5 m. Offline outputs are window-level; invalid windows have no prediction.',
                    sha256={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
        (root/'release.json').write_text(json.dumps(info, indent=2))
        root.rename(out)
    verify_bundle(out)


def upload(root, repo):
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    import os
    load_dotenv(); verify_bundle(root)
    if repo.count('/') != 1:
        raise ValueError('Use namespace/repo-name')
    api = HfApi(token=os.getenv('HF_TOKEN') or True)
    # Dedicated fresh repository: never update a public or unrelated model.
    api.create_repo(repo, repo_type='model', private=True, exist_ok=False)
    if not api.model_info(repo).private:
        raise RuntimeError('Refusing public upload')
    allowed = [*json.loads((root/'release.json').read_text())['sha256'], 'release.json']
    api.upload_folder(repo_id=repo, repo_type='model', folder_path=root, allow_patterns=allowed)


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd',required=True)
    pack = sub.add_parser('pack'); publish = sub.add_parser('publish')
    for key in ('data','classification','distance','out'):
        pack.add_argument('--'+key,required=True,type=Path)
    publish.add_argument('--bundle',required=True,type=Path); publish.add_argument('--repo',required=True)
    a=ap.parse_args()
    if a.cmd=='pack': build(a.data,a.classification,a.distance,a.out)
    else: upload(a.bundle,a.repo)


if __name__=='__main__': main()
