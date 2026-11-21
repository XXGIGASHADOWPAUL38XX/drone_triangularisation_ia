# EDTH — détection de drone et distance par téléphone (1–5 m, en main)

Pipeline en étapes numérotées, chacune dans son dossier avec ses données
entrantes (`input/`) et sortantes (`output/`) : on peut inspecter, copier ou
remplacer les fichiers entre deux étapes à la main. Les architectures sont
gelées d'après l'HPO préliminaire (run RTX 3090) ; ce repo sert à les
re-entraîner proprement et à expérimenter autour (sample rate, features)
sans casser la méthodologie.

## Le repo en une image

```
00_download_dataset/   corpus (local ou HF privé)      -> output/data/samples/
01_ppc_features/       fenêtres + features + splits     -> output/windows-<rate>.parquet
02_ppc_classification/ détecteur (CNN gelé, baselines)  -> output/<run>/ (modèle + métriques)
03_ppc_distance/      distance (MLP gelé, baselines)  -> output/<run>/ (modèle + métriques)
04_compare/            tableau de tous les runs         -> output/comparatif.csv
ml.py                  bibliothèque partagée de 02 et 03 (modèles, métriques, expérience)
run_server.sh          tout enchaîne sur la machine RTX
tests/                 parité 24 kHz, fuite de groupes, smoke des modèles
```

Chaque étape copie automatiquement sa sortie vers l'entrée de la suivante ;
le contenu de `input/` et `output/` n'est pas versionné (ce sont des données).

```
WAV maîtres (mono PCM16 48 kHz)
   │  00  (corpus local ou HF privé)
   ▼ 01_ppc_features/input/data ──> 01 ──> output/windows-<rate>.parquet + splits.json
   │        (FIR anti-repliement, contrôle qualité, fenêtres 1 s / pas 250 ms)
   ▼ copié automatiquement vers 02/input et 03/input
02 / 03 : CV groupée sur dev (3 plis) ──> réajustement final ──> UNE lecture du holdout
   ▼
output/<run>/ : model.joblib, metrics.json, config.json, cv.csv, curves.json, courbes
   │  04_compare
   ▼
04_compare/output/comparatif.csv
```

## Ce qui est gelé d'après l'HPO préliminaire

| Tâche | Modèle retenu | Entrée | Dev (CV groupée) | Holdout |
|---|---|---|---|---|
| Détection | **CNN 2D** : 1 bloc conv 32 canaux, dropout 0.15, ~450 paramètres | log-mel 32×22 **relative**, sans one-hot téléphone | 97,4 % balanced acc. | 96,5 % |
| Distance | **MLP** : 16 cachées, dropout 0.15, ~577 paramètres, sortie [1, 5] m | 34 niveaux **absolus** (32 bandes mel + 4–8 kHz + RMS), sans one-hot | MAE 0,41–0,43 m | 0,50 m |

Baselines à refaire à côté : logistique relative (90,8 % holdout), ridge
(~0,59 m), constant 3 m (1,09 m). Rejetés par l'HPO : one-hot téléphone,
bande 4–8 kHz seule, calibration log-distance à 1 m.

## Démarrage rapide

Python 3.12 ; PyTorch s'installe à part (index CPU ou CUDA) :

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m pytest -q     # 15 tests, dont la parité 24 kHz contre le cache historique
```

```sh
python 00_download_dataset/main.py --local /home/noe/Projects/EDTH_data-collector/data
# ou : python 00_download_dataset/main.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>

python 01_ppc_features/main.py --rate 24000        # défaut ; 16000 et 48000 supportés

python 02_ppc_classification/main.py               # CNN gelé -> output/cnn-24k/
python 02_ppc_classification/main.py --model logistic --relative --name logit
python 03_ppc_distance/main.py                    # MLP gelé -> output/mlp-24k/
python 03_ppc_distance/main.py --model constant && python 03_ppc_distance/main.py --model ridge

python 04_compare/main.py
```

Sur la machine RTX : `bash run_server.sh` (deux modèles gelés + les trois
baselines + tableau ; plus d'Optuna, quelques minutes au total).

## Comment on expérimente

Principe : **une expérience = une commande 02 ou 03 = un dossier
`output/<nom>`**, avec ses features et ses hyperparamètres enregistrés dans
`config.json` et visibles dans le tableau de 04. On choisit entre
expériences sur la colonne `dev_cv` — jamais sur le holdout.

### Sample rate

Les features dépendent du taux, **pas les splits** : `splits.json` est écrit
une fois puis réutilisé (l'étape 01 le recopie dans son propre `input/`),
donc tous les taux comparent sur les mêmes prises et les mêmes plis.

```sh
python 01_ppc_features/main.py --rate 16000        # idem --rate 48000
# puis, pour tester ce taux :
python 02_ppc_classification/main.py --input input/windows-16000.parquet --name cnn-16k
python 03_ppc_distance/main.py      --input input/windows-16000.parquet --name mlp-16k
python 04_compare/main.py
```

Questions : 24 kHz perd-il quelque chose face à 48 ? 16 kHz (Nyquist 8 kHz)
rogne-t-il la bande utile 4–8 kHz et casse-t-il la distance ?

### Mixer des features (mlp, logistic, ridge)

| Groupe | Colonnes | Ce que teste l'expérience |
|---|---|---|
| `mel` | les 32 bandes mel | le spectre complet |
| `b48` | seulement 4–8 kHz | l'hypothèse « bande du drone » seule |
| `rms` | seulement le niveau global | le niveau absolu, sans spectre |
| *(vide)* | les 34 colonnes | l'entrée gelée du MLP distance |

```sh
python 03_ppc_distance/main.py --model mlp --features b48 --name mlp-b48
python 03_ppc_distance/main.py --model mlp --features rms --name mlp-rms
python 02_ppc_classification/main.py --model mlp --features mel,b48 --name mlp-mel-b48
```

Le CNN n'a pas d'option features : il est gelé sur le log-mel complet.

### Niveau relatif vs absolu

`--relative` retire le niveau global de chaque fenêtre avant le modèle
(moins sensible au volume émis). C'est le choix gelé du CNN — et une
expérience à part pour mlp/logistic/ridge. Pour la distance, les niveaux
ABSOLUS sont la convention : `--relative` y est contre-indiqué.

### Hyperparamètres

`--lr` et `--dropout` surchargent les valeurs gelées ; c'est un choix
délibéré, visible dans `config.json` et dans le tableau de 04.

### La règle d'or

Le holdout n'est lu qu'à la fin d'un run 02/03, **une seule fois par
question**, après avoir choisi sur `dev_cv`. Chaque variante supplémentaire
évaluée sur le holdout l'affaiblit. Les 9 748 fenêtres se chevauchent : ce
ne sont pas 9 748 observations indépendantes.

## Sorties d'un run (`02/output/<nom>/` ou `03/output/<nom>/`)

- `model.joblib` — modèle + scaler + colonnes, rechargeable (joblib) ;
- `metrics.json` — dev_cv, holdout (overall, par téléphone, par distance
  cible), MAE non bornée pour la distance ;
- `config.json` — modèle, features, hyperparamètres, taux, empreintes ;
- `cv.csv`, `curves.json`, `learning-curves.png` — plis et apprentissage.

## Méthodologie (à ne pas casser)

Mêmes règles que la spécification ML V1 du hackathon : répartition par
**événement temporel avant fenêtrage** (un groupe ne traverse jamais
dev/test), **poids égal par prise**, scaler ajusté sur le train du pli
uniquement, fenêtres valides de 1 s / pas 250 ms après contrôle qualité
(suites de zéros ≥ 40 ms, garde 500 ms), niveaux **absolus** en dBFS pour la
distance. La parité 24 kHz avec le cache qui a servi au run RTX est testée
automatiquement à chaque `pytest` (régénération complète, comparaison
bit-à-bit des 9 748 fenêtres et des splits).

Limites à dire telles quelles devant le jury : même campagne, mêmes trois
appareils, une seule prise à 5 m par téléphone ; ni précision 2D, ni latence
de détection, ni garantie à ±50 cm par mesure.

## Où sont passés les anciens fichiers ?

Dans l'historique Git (`b2e845d` et commits précédents) : la version plate
à la racine, le package `edth/` et les dossiers `00_`–`03_` d'origine. Les
noms d'étapes et la mécanique `input/`/`output/` reprennent celles du commit
`861bd9c` ; seule la logique scientifique a changé (24 kHz, fenêtres 1 s,
features absolues, splits groupés, modèles gelés).
