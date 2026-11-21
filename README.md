# EDTH — détection de drone et distance par téléphone (1–5 m, en main)

Pipeline volontairement plat et court : tout est à la racine, quatre étapes
numérotées à lancer dans l'ordre. Les résultats préliminaires (Optuna sur
RTX 3090) ont déjà tranché les architectures ; ce repo sert à les re-entraîner
proprement et à expérimenter autour (sample rate, features) sans casser la
méthodologie.

## Ce qui est gelé d'après l'HPO préliminaire

| Tâche | Modèle retenu | Entrée | Dev (CV groupée) | Holdout |
|---|---|---|---|---|
| Détection présent/absent | **CNN 2D** : 1 bloc conv 32 canaux, dropout 0.15, ~450 paramètres | log-mel 32×22 **relative** (niveau global retiré), sans one-hot téléphone | 97,4 % balanced acc. | 96,5 % |
| Distance (si présent) | **MLP** : 16 cachées, dropout 0.15, ~577 paramètres | 34 niveaux **absolus** (32 bandes mel + 4–8 kHz + RMS), sans one-hot | MAE 0,41–0,43 m | 0,50 m |

Baselines de contexte à refaire à côté : logistique relative (90,8 % holdout),
ridge pleine bande (~0,56–0,59 m), constant 3 m (1,09 m). Rejetés par l'HPO :
one-hot téléphone, bande 4–8 kHz seule, calibration log-distance à 1 m.

## Démarrage rapide

Python 3.12. PyTorch s'installe à part (index CPU ou CUDA) :

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m pytest -q          # 14 tests, dont la parité 24 kHz contre le cache historique
```

```sh
# 0. Cache de features + splits groupés (une fois ; ~5 s sur le corpus pilote)
python 00_prepare.py --local /home/noe/Projects/EDTH_data-collector/data
#    ou depuis le Hub privé : python 00_prepare.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>

# 1. Entraînement (une commande = une expérience ; le holdout n'est pas lu)
python 01_train.py --task classification --model cnn --out runs/cls-24k
python 01_train.py --task distance      --model mlp --out runs/dist-24k

# 2. Holdout (une seule consultation par question !)
python 02_evaluate.py --run runs/cls-24k
python 02_evaluate.py --run runs/dist-24k

# 3. Tableau comparatif de tous les runs
python 03_compare.py
```

Sur la machine RTX : `bash run_server.sh` (entraîne les deux modèles gelés,
évalue, compare).

## Experimenter sans se tirer dans le pied

- **Sample rate** : `python 00_prepare.py --local ... --rate 16000` (ou 48000)
  crée `artifacts/windows-<rate>.parquet`. Les **splits restent les mêmes**
  (`artifacts/splits.json`, créé au premier appel) : toutes les comparaisons de
  taux se font sur des partitions identiques. Question ouverte : 24 kHz
  suffit-il face à 48, et 16 kHz casse-t-il la bande utile 4–8 kHz ?
- **Features** : les modèles plats prennent `--features mel,b48,rms`
  (groupes combinables : `mel` = 32 bandes, `b48` = bande 4–8 kHz,
  `rms` = niveau global ; vide = tout). Le CNN, lui, est gelé sur le log-mel.
- **Niveau relatif** : `--relative` pour logistique/mlp/ridge ; le CNN est
  relatif par conception (`--no-relative` pour le défier).
- **Choisir sur le dev_cv de 03_compare.py**, jamais sur le holdout : chaque
  variante supplémentaire évaluée sur `test` l'affaiblit. Les 9 748 fenêtres
  se chevauchent : ce ne sont pas 9 748 observations indépendantes.

## Méthodologie (à ne pas casser)

Mêmes règles que la spécification ML V1 du hackathon : répartition par
**événement temporel avant fenêtrage** (un groupe ne traverse jamais
dev/test), **poids égal par prise**, scaler ajusté sur le train du pli
uniquement, fenêtres valides de 1 s / pas 250 ms après contrôle qualité
(coupures numériques ≥ 40 ms, garde 500 ms), niveaux **absolus** en dBFS pour
la distance. La parité 24 kHz avec le cache qui a servi au run RTX est
testée automatiquement (`tests/test_pipeline.py`). Les hyperparamètres par
défaut sont les valeurs retenues par l'HPO ; les changer est un choix
délibéré, pas un réglage caché.

Limites à dire telles quelles : même campagne, mêmes trois appareils, une
seule prise à 5 m par téléphone ; ni précision 2D, ni latence de détection,
ni garantie à ±50 cm par mesure.

## Sorties d'un run

`model.joblib` (modele + scaler + vocabulaire), `config.json` (params,
empreintes du cache et du code), `cv.csv` (score par pli), `curves.json`,
`learning-curves.png` et — après `02_evaluate.py` — `metrics.json`
(global, par téléphone, par distance). `03_compare.py` agrège tout en un
tableau.
