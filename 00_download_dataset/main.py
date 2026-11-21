import os
from pathlib import Path
from shutil import copytree

from datasets import load_dataset
from dotenv import load_dotenv

STAGE = Path(__file__).resolve().parent
OUTPUT = STAGE / "output/train"
FEATURES_INPUT = STAGE.parent / "01_ppc_features/input/train"


def download_dataset():
    load_dotenv(STAGE.parent / ".env")
    return load_dataset(os.getenv("REPO"), token=os.getenv("TOKEN"), split="train")


def main():
    dataset = download_dataset()
    dataset.save_to_disk(str(OUTPUT))
    copytree(OUTPUT, FEATURES_INPUT, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
