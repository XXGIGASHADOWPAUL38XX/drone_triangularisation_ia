# EDTH — détection de drone et distance par téléphone (1–7 m, en main)

Pipeline en étapes numérotées, chacune dans son dossier avec ses données
entrantes (`input/`) et sortantes (`output/`) : on peut inspecter, copier ou
remplacer les fichiers entre deux étapes à la main. Les architectures sont
gelées d'après l'HPO préliminaire (run RTX 3090) ; ce repo sert à les
re-entraîner proprement et à expérimenter autour (sample rate, features)
sans casser la méthodologie.

## Le repo en une image

```
00_download_dataset/   corpus (local ou HF privé)      -> output/data/ (samples/ ou data/)
01_ppc_features/       fenêtres + features + splits    -> deux caches parquet pour 02 et 03
02_ppc_classification/ détecteur (MLP, baselines)        -> output/<run>/ (modèle + métriques)
03_ppc_distance/      distance (MLP gelé, baselines)  -> output/<run>/ (modèle + métriques)
04_compare/            tableau de tous les runs         -> output/comparatif.csv
05_export_release/     fige 2 runs -> bundle de release  -> output/<release>/ (contrat edth.model-release.v2)
ml.py                  bibliothèque partagée de 02 et 03 (modèles, métriques, expérience)
run_server.sh          tout enchaîne sur la machine RTX
tests/                 parité 24 kHz, fuite de groupes, smoke des modèles
```

Chaque étape copie automatiquement sa sortie vers l'entrée de la suivante ;
le contenu de `input/` et `output/` n'est pas versionné (ce sont des données).

```
WAV maîtres (mono PCM16 48 kHz)
   │  00  (corpus local ou HF privé)
   ▼ 01_ppc_features/input/data ──> 01 ──> parquet de features + splits.json
   │        (48 kHz natif, contrôle qualité, fenêtres 100 ms / pas 50 ms pour 02 et 20 ms pour 03)
   ▼ copié automatiquement vers 02/input et 03/input
02 / 03 : entraînement unique par groupes ──> validation ──> holdout
   ▼
output/<run>/ : model.joblib, metrics.json, config.json, curves.json, courbes
   │  04_compare
   ▼
04_compare/output/comparatif.csv
```

## Modèles et entrées

| Tâche | Modèle | Entrée actuelle | Dev historique | Holdout historique |
|---|---|---|---|---|
| Détection | **MLP**, 16 unités cachées | 50 niveaux de bandes | 97,5 % balanced acc. (CNN) | 96,2 % (CNN) |
| Distance | **MLP**, 16 unités cachées, sortie [1, 7] m | 50 niveaux de bandes | MAE 0,43 m | 0,46 m |

Les scores historiques correspondent aux anciennes entrées. Régénérer les parquets à l'étape 01 et réentraîner les modèles pour évaluer les entrées FFT.

Baselines à refaire à côté : logistique relative (90,8 % holdout), ridge
(~0,59 m), constant 3 m (1,09 m). Rejetés par l'HPO : one-hot téléphone,
bande 4–8 kHz seule, calibration log-distance à 1 m. Chiffres du tableau :
reproduction locale (CPU) du protocole gelé ; la RTX 3090 donne le même
protocole à 97,4 % / 96,5 % (bruit flottant GPU attendu).

## Démarrage rapide

Python 3.12 ; PyTorch s'installe à part (index CPU ou CUDA) :

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m pytest -q
```

```sh
python 00_download_dataset/main.py --local /home/noe/Projects/EDTH_data-collector/data
# ou : python 00_download_dataset/main.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>

python 01_ppc_features/main.py

python 02_ppc_classification/main.py
python 02_ppc_classification/main.py --model logistic --name logit
python 03_ppc_distance/main.py
python 03_ppc_distance/main.py --model constant && python 03_ppc_distance/main.py --model ridge

python 04_compare/main.py

# figer les runs choisis en une release (contrat d'intégration v2, doc Outline B2)
python 05_export_release/main.py --push     # bundle autovalidé + push vers HF_MODELS_REPO
```

La release lit la fenêtre, la géométrie des bandes et la recette de recalcul
des features dans le manifeste de l'étape 01 — les deux runs doivent partager
le même cache. Seules les features LOCALES de fenêtre sont exportables : les
descripteurs normalisés par prise (`handcrafted`, band descriptors) ne sont
pas rejouables en temps réel et sont refusés.

Sur la machine RTX : `bash run_server.sh` (deux modèles gelés + les trois
baselines + tableau ; plus d'Optuna, quelques minutes au total).

## Comment on expérimente

Principe : **une expérience = une commande 02 ou 03 = un dossier
`output/<nom>`**, avec ses features et ses hyperparamètres enregistrés dans
`config.json` et visibles dans le tableau de 04. On choisit entre
expériences sur la colonne `validation`. La colonne `dev_cv` correspond aux runs historiques.

### Sample rate

Les features dépendent du taux, **pas les splits** : `splits.json` est écrit
une fois puis réutilisé (l'étape 01 le recopie dans son propre `input/`),
donc tous les taux comparent les mêmes partitions d'entraînement, de validation et de test.
Le fichier est **versionné** (`01_ppc_features/input/splits.json`) : une machine
fraîche le récupère par `git pull`, et l'étape 01 doit afficher
« partitions réutilisées » — sinon les résultats ne sont plus comparables.
Les anciens splits gardent leur holdout et réutilisent le pli 0 comme validation.
Le pipeline et Optuna utilisent par défaut 48 kHz et une fenêtre de 100 ms.
La classification utilise un pas de 50 ms (`windows-48000-bands200-avg4-w100-s50.parquet`).
La distance utilise un pas de 20 ms (`windows-48000-bands200-avg4-w100-s20.parquet`).
Les options `--window-ms` et `--stride-ms` règlent la durée et le pas en millisecondes.
Le pas accepte des valeurs fractionnaires, par exemple `--stride-ms 62.5`.
La fenêtre doit contenir au moins les 2 048 échantillons d'une trame FFT.
Pour conserver plusieurs caches à un même taux, donner un suffixe avec
`--cache-tag`, par exemple `--cache-tag w100-s50`.

```sh
python 01_ppc_features/main.py --rate 16000 --window-ms 250 --stride-ms 125
# puis, pour tester ce taux (depuis la racine) :
python 02_ppc_classification/main.py --input 02_ppc_classification/input/windows-16000-bands200-avg4.parquet --features band_levels_avg4 --name mlp-16k
python 03_ppc_distance/main.py      --input 03_ppc_distance/input/windows-16000-bands200-avg4.parquet --name mlp-16k
python 04_compare/main.py
```

Les expériences comparent notamment l'effet de la fréquence maximale observable
sur la détection et l'estimation de distance.

### Captures WebKit (iPhone)

L'étape 01 n'accepte par défaut que le contrat `edth.web-pcm.v1` : les prises
iPhone (contrat `edth.web-pcm.webkit.v1`, profil non validé côté app) sont
exclues du corpus. L'option `--webkit` les intègre — les caches produits
portent un suffixe `-webkit` et se comparent aux caches stricts **avec les
mêmes partitions**. Générer l'ancre avec `--webkit` pour couvrir tout le
corpus (sinon les prises nouvelles restent `excluded`) :

```sh
rm 01_ppc_features/input/splits.json
python 01_ppc_features/main.py --webkit   # ancre + caches -webkit (s20 et s50)
python 01_ppc_features/main.py            # caches stricts, mêmes partitions
```

### Mixer des features (mlp, logistic, ridge)

| Groupe | Colonnes | Ce que teste l'expérience |
|---|---|---|
| `band_levels_avg4` | 50 niveaux de bandes en dBFS | le spectre regroupé |
| `rms` | seulement le niveau global | le niveau absolu, sans spectre |
| `handcrafted` | six descripteurs de la fenêtre entière | la forme d'onde et le spectre |
| Sans `--features` | 50 niveaux de bandes | le groupe par défaut `band_levels_avg4` |

Les descripteurs sont normalisés par prise. L'entropie, le centroïde, le flux
et l'onset utilisent le spectre entier. Le taux de
passage par zéro et le facteur de crête utilisent la forme d'onde.

```sh
python 01_ppc_features/main.py --with-descriptors
python 03_ppc_distance/main.py --input 03_ppc_distance/input/windows-48000-bands200-avg4-descriptors-w100-s20.parquet --features rms --name mlp-rms
python 03_ppc_distance/main.py --input 03_ppc_distance/input/windows-48000-bands200-avg4-descriptors-w100-s20.parquet --features handcrafted --name mlp-handcrafted
python 02_ppc_classification/main.py --input 02_ppc_classification/input/windows-48000-bands200-avg4-descriptors-w100-s20.parquet --features band_levels_avg4,rms --name mlp-bands-rms
```

L'option `--full-spectrum` de l'étape 01 régénère le cache FFT des anciens
essais. Le modèle `--model cnn` utilise la colonne `logfft` de ce cache.

Pour tester la puissance 4–8 kHz, lancer l'étape 01 avec `--include-b48` puis
sélectionner `windows-48000-b48-w100-s20.parquet` avec `--input` et ajouter `b48`
à `--features`.

Pour entraîner sur les 100 niveaux de bandes FFT, lancer l'étape 01 avec
`--band-levels`. Sélectionner `windows-48000-bands100-w100-s20.parquet` avec `--input`
et utiliser `--features band_levels` avec `--model mlp` pour la classification.

`--band-levels-avg3` calcule 200 niveaux en dBFS et produit 67 features avec
`--features band_levels_avg3`. Les deux dernières bandes forment le dernier groupe.

`--band-levels-avg4` calcule 200 niveaux en dBFS et moyenne chaque groupe de
quatre bandes voisines. Le cache par défaut `windows-48000-bands200-avg4-w100-s20.parquet` contient
les 50 features sélectionnées avec `--features band_levels_avg4`.
Ajouter `--with-descriptors` produit un cache de 57 features.
`--with-band-descriptors` ajoute plutôt l'entropie, le centroïde, le flux et
l'onset de chaque bande avg4. Sélectionner `band_levels_avg4,band_descriptors_avg4`
pour entraîner sur les 250 features de ce cache.
`--band-levels-avg5` moyenne les niveaux par groupes de cinq et produit 40
features sélectionnées avec `--features band_levels_avg5` dans
`windows-<rate>-bands200-avg5.parquet`.
`--band-levels-avg6` moyenne 300 bandes par groupes de six et produit 50
features sélectionnées avec `--features band_levels_avg6` dans
`windows-<rate>-bands300-avg6.parquet`.

### Niveau relatif vs absolu

`--relative` retire le niveau global de chaque fenêtre avant le modèle
(moins sensible au volume émis). C'est le choix gelé du CNN — et une
expérience à part pour mlp/logistic/ridge. Pour la distance, les niveaux
ABSOLUS sont la convention : `--relative` y est contre-indiqué.

### Hyperparamètres

`--lr` et `--dropout` surchargent les valeurs gelées ; c'est un choix
délibéré, visible dans `config.json` et dans le tableau de 04.

### Validation et test

La validation choisit le point d'arrêt et le holdout mesure le modèle final.
Le classement s'appuie sur la validation. Le chevauchement rend les fenêtres
d'une même prise dépendantes.

## Sorties d'un run (`02/output/<nom>/` ou `03/output/<nom>/`)

- `model.joblib` — modèle + scaler + colonnes, rechargeable (joblib) ;
- `metrics.json` — validation, holdout (overall, par téléphone, par distance
  cible), MAE non bornée pour la distance ;
- `config.json` — modèle, features, hyperparamètres, taux, empreintes ;
- `curves.json`, `learning-curves.png` — apprentissage et arrêt anticipé.

## Méthodologie (à ne pas casser)

Mêmes règles que la spécification ML V1 du hackathon : répartition par
**événement temporel** (chaque groupe appartient à une partition),
**poids égal par prise**, scaler ajusté sur les données d'entraînement
uniquement, fenêtres valides de 250 ms / pas 125 ms par défaut après contrôle qualité
(suites de zéros ≥ 40 ms, garde 500 ms), niveaux **absolus** en dBFS pour la
distance. Avec le cache historique disponible, les tests comparent les fenêtres,
les partitions et le niveau RMS à 24 kHz avec les anciens réglages
de 1 000 ms et 250 ms.

Limites à dire telles quelles devant le jury : même campagne, mêmes trois
appareils, une seule prise à 5 m par téléphone ; ni précision 2D, ni latence
de détection, ni garantie à ±50 cm par mesure.

## Où sont passés les anciens fichiers ?

Dans l'historique Git (`b2e845d` et commits précédents) : la version plate
à la racine, le package `edth/` et les dossiers `00_`–`03_` d'origine. Les
noms d'étapes et la mécanique `input/`/`output/` reprennent celles du commit
`861bd9c` ; seule la logique scientifique a changé (24 kHz, fenêtres 1 s,
features absolues, splits groupés, modèles gelés).
