"""Étape 0 : rassembler le corpus (WAV maîtres 48 kHz + métadonnées).

  python main.py --local /home/noe/Projects/EDTH_data-collector/data
  python main.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>

La sortie output/data/ (samples/ en local, AudioFolder data/ via HF)
alimente l'entrée de l'étape 01.
"""
import argparse
import os
from pathlib import Path
from shutil import copytree

from dotenv import load_dotenv

STAGE = Path(__file__).resolve().parent
OUT_DATA = STAGE / "output" / "data"  # samples/ (local) ou data/ (AudioFolder HF)
NEXT_INPUT = STAGE.parent / "01_ppc_features" / "input"


def main():
    load_dotenv(STAGE.parent / ".env", override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, help="dossier data du collecteur (contient samples/)")
    p.add_argument("--repo", default=os.getenv("HF_DATASET"), help="dataset HF privé")
    p.add_argument("--revision", default=os.getenv("HF_REVISION"))
    a = p.parse_args()
    if not a.local and not a.repo:
        p.error("préciser --local ou --repo")
    if a.local:
        src = a.local / "samples" if (a.local / "samples").is_dir() else a.local
        copytree(src, OUT_DATA / "samples", dirs_exist_ok=True)
        n = len(list((OUT_DATA / "samples").glob("*/audio.wav")))
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(a.repo, revision=a.revision, repo_type="dataset",
                                      token=os.getenv("HF_TOKEN") or True, allow_patterns=["data/*"]))
        copytree(root, OUT_DATA, dirs_exist_ok=True)  # AudioFolder : data/<id>.wav + data/metadata.csv
        n = len(list((OUT_DATA / "data").glob("*.wav")))
    print(f"{n} prises dans {OUT_DATA}")
    copytree(OUT_DATA, NEXT_INPUT / "data", dirs_exist_ok=True)
    print(f"copié vers {NEXT_INPUT / 'data'}")


if __name__ == "__main__":
    main()
