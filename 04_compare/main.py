"""Étape 4 : tableau comparatif de tous les runs des étapes 02 et 03.

  python main.py            # lit 02/output et 03/output, écrit output/comparatif.csv

Chaque ligne = un run (une expérience). La colonne dev_cv sert à CHOISIR ;
la colonne test ne fait que confirmer un choix déjà fait sur le dev, elle ne
sert pas à en tester de nouveaux.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent
OUTPUT = STAGE / "output"
STEPS = ("02_ppc_classification", "03_ppc_distance")
TEST_KEY = {"classification": "test_balanced_accuracy", "distance": "test_mae_m"}


def collect(steps):
    rows = []
    for step in steps:
        for config_path in sorted((STAGE.parent / step / "output").glob("*/config.json")):
            run = config_path.parent
            config = json.loads(config_path.read_text())
            metric = "balanced_accuracy" if config["task"] == "classification" else "mae_m"
            row = dict(run=run.name, task=config["task"], model=config["model"],
                       features=config["features"], rate=config.get("rate"),
                       relative=config.get("relative"), dev_cv=config["dev_cv"][metric])
            metrics_path = run / "metrics.json"
            if metrics_path.is_file():
                row[TEST_KEY[config["task"]]] = json.loads(metrics_path.read_text())["test"]["overall"][metric]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", nargs="*", default=list(STEPS))
    a = p.parse_args()
    table = collect(a.steps)
    if table.empty:
        raise SystemExit("aucun run trouvé : lancer les étapes 02 et 03 d'abord.")
    # détection : plus haut = mieux ; distance : plus bas = mieux
    table = (table.assign(_sort=lambda t: t.dev_cv.where(t.task == "distance", -t.dev_cv))
                  .sort_values(["task", "_sort"]).drop(columns="_sort"))
    print(table.to_string(index=False))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "comparatif.csv", index=False)
    print(f"écrit : {OUTPUT / 'comparatif.csv'}")


if __name__ == "__main__":
    main()
