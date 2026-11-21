# EDTH — détection de drone et distance par téléphone (1–5 m, en main)

Pipeline volontairement plat : tout est à la racine, quatre étapes numérotées
à lancer dans l'ordre. Les résultats préliminaires (Optuna sur RTX 3090) ont
déjà tranché les architectures ; ce repo sert à les re-entraîner proprement et
à expérimenter autour (sample rate, features) sans casser la méthodologie.

## Architecture du repo — qui sert à quoi

### Les briques (modules importés, jamais lancés directement)

| Fichier | Rôle |
|---|---|
| [audio.py](<audio.py>) | Lit les WAV maîtres stricts (mono PCM16 48 kHz), filtre anti-repliement et décime vers le taux cible (FIR à état, 24 kHz = référence historique exacte), contrôle qualité (suite de zéros ≥ 40 ms, garde 500 ms), découpe en fenêtres valides de 1 s / pas 250 ms. |
| [features.py](<features.py>) | Calcule les features d'une fenêtre : 32 niveaux de bandes mel (`f0..f31`), bande 4–8 kHz (`b4_8`), niveau global (`rms_db`), spectrogramme log-mel (`logmel`, 32 × trames). Définit les groupes nommés `mel`/`b48`/`rms` utilisables au CLI. |
| [models.py](<models.py>) | Les quatre modèles : `CnnPresence` (gelé), `Mlp`, `Linear` (logistique/ridge), `Constant`. Contient aussi la pondération « une masse égale par prise ». Les défauts d'hyperparamètres sont les valeurs retenues par l'HPO. |
| [data.py](<data.py>) | Charge le cache parquet et vérifie son intégrité (aucun groupe ne traverse dev/test, étiquettes cohérentes), fournit les métriques pondérées par prise, globalement / par téléphone / par distance. |

### Les étapes numérotées (le pipeline, dans l'ordre)

| Étape | Entrée → Sortie | Lit le holdout ? |
|---|---|---|
| [00_prepare.py](<00_prepare.py>) | Dossier du collecteur (ou repo HF privé, `--repo`) → `artifacts/windows-<rate>.parquet` + manifeste `.json` + `artifacts/splits.json`. ~5 s sur le corpus pilote. | — |
| [01_train.py](<01_train.py>) | Un cache + une tâche + un modèle → `runs/<nom>/` : `model.joblib`, `config.json`, `cv.csv`, `curves.json`. Validation croisée sur les 3 plis de dev, puis réajustement final sur tout dev. | **Non** |
| [02_evaluate.py](<02_evaluate.py>) | Un run + son cache → `metrics.json` (global, par téléphone, par distance) + `learning-curves.png`. | **Oui (c'est son seul travail)** |
| [03_compare.py](<03_compare.py>) | Tous les `runs/*/` → un tableau : dev_cv par défaut, holdout si évalué. Comparer sans rien relancer. | lit, ne choisit pas |

Autres fichiers : [run_server.sh](<run_server.sh>) enchaîne les deux modèles
gelés sur la RTX puis évalue et compare ; [tests/test_pipeline.py](<tests/test_pipeline.py>)
(14 tests) garde la parité 24 kHz bit-à-bit contre le cache historique, la
garde qualité, l'absence de fuite de groupes et un smoke de chaque modèle.

### Le flux de données en une image

```
WAV maîtres (collecteur, mono PCM16 48 kHz)
        │  00_prepare.py  --rate 16000|24000|48000
        ▼
artifacts/windows-<rate>.parquet     artifacts/splits.json
(un jeu de features par taux)        (créé UNE fois, partagé par tous les taux)
        │  01_train.py   ← une commande = une expérience
        ▼
runs/<nom>/ {model.joblib, config.json, cv.csv, curves.json}
        │  02_evaluate.py   ← holdout, une seule consultation par question
        ▼
runs/<nom>/ {metrics.json, learning-curves.png}
        │  03_compare.py
        ▼
tableau comparatif de toutes les expériences
```

## Ce qui est gelé d'après l'HPO préliminaire

| Tâche | Modèle retenu | Entrée | Dev (CV groupée) | Holdout |
|---|---|---|---|---|
| Détection présent/absent | **CNN 2D** : 1 bloc conv 32 canaux, dropout 0.15, ~450 paramètres | log-mel 32×22 **relative** (niveau global retiré), sans one-hot téléphone | 97,4 % balanced acc. | 96,5 % |
| Distance (si présent) | **MLP** : 16 cachées, dropout 0.15, ~577 paramètres | 34 niveaux **absolus** (32 bandes mel + 4–8 kHz + RMS), sans one-hot | MAE 0,41–0,43 m | 0,50 m |

Baselines de contexte à refaire à côté : logistique relative (90,8 % holdout),
ridge pleine bande (~0,56–0,59 m), constant 3 m (1,09 m). Rejetés par l'HPO :
one-hot téléphone, bande 4–8 kHz seule pour le MLP, calibration log-distance
à 1 m.

## Démarrage rapide

Python 3.12. PyTorch s'installe à part (index CPU ou CUDA) :

```sh
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -r requirements.txt
python -m pytest -q          # 14 tests, dont la parité 24 kHz contre le cache historique
```

```sh
# 0. Cache de features + splits groupés (une fois)
python 00_prepare.py --local /home/noe/Projects/EDTH_data-collector/data
#    ou depuis le Hub privé : python 00_prepare.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>

# 1. Entraînement des deux modèles gelés
python 01_train.py --task classification --model cnn --out runs/cls-24k
python 01_train.py --task distance      --model mlp --out runs/dist-24k

# 2. Holdout
python 02_evaluate.py --run runs/cls-24k
python 02_evaluate.py --run runs/dist-24k

# 3. Tableau
python 03_compare.py
```

Sur la machine RTX : `bash run_server.sh` fait les étapes 1→3 pour les deux
modèles gelés (plus d'Optuna : quelques minutes).

## Comment on expérimente

Principe unique : **une expérience = une commande `01_train.py` = un dossier
`runs/<nom>`**. Tout ce qui distingue l'expérience (cache, features,
hyperparamètres) est enregistré dans son `config.json` et ressort dans le
tableau de `03_compare.py`. On choisit entre expériences sur la colonne
`dev_cv` — jamais sur le holdout.

### 1. Changer de sample rate

Les features dépendent du taux ; les **splits, non** : `artifacts/splits.json`
est créé au premier `00_prepare.py` puis réutilisé, donc deux taux comparent
sur exactement les mêmes prises et les mêmes plis.

```sh
python 00_prepare.py --local ... --rate 16000        # idem --rate 48000
python 01_train.py --task classification --model cnn --cache artifacts/windows-16000.parquet --out runs/cls-16k
python 01_train.py --task distance      --model mlp --cache artifacts/windows-16000.parquet --out runs/dist-16k
python 03_compare.py
```

Questions auxquelles ça répond : 24 kHz perd-il quelque chose face à 48 ?
16 kHz (Nyquist 8 kHz) rogne-t-il la bande utile 4–8 kHz et casse-t-il la
distance ?

### 2. Mixer des features (modèles plats : mlp, logistic, ridge)

| Groupe | Colonnes | Ce que teste l'expérience |
|---|---|---|
| `mel` | les 32 bandes mel | le spectre complet |
| `b48` | seulement 4–8 kHz | l'hypothèse « bande du drone » seule |
| `rms` | seulement le niveau global | le niveau absolu, sans information spectrale |
| *(vide)* | les 34 colonnes | l'entrée gelée du MLP distance |

```sh
python 01_train.py --task distance --model mlp --features b48        --out runs/dist-b48
python 01_train.py --task distance --model mlp --features rms        --out runs/dist-rms
python 01_train.py --task distance --model mlp --features mel,b48   --out runs/dist-mel-b48
```

Le CNN n'a pas d'option features : il est gelé sur le log-mel complet
(c'est son architecture retenue).

### 3. Niveau relatif vs absolu

`--relative` retire le niveau global de chaque fenêtre avant le modèle
(moins sensible au volume émis) ; c'est le choix gelé du CNN, et une
expérience à part pour les modèles plats. Le CNN reste relatif par
conception : `--no-relative` pour le mettre à l'épreuve.

### 4. Toucher aux hyperparamètres (à froid)

`--lr`, `--dropout`, `--width`, `--depth`, `--strength`, `--epochs`,
`--patience` surchargent les valeurs gelées. C'est un choix délicat et
visible : il figure dans `config.json` et dans `03_compare.py`, personne ne
le fera sans le savoir.

### 5. Rappeler les baselines à côté

Une feature n'est « bonne » que si elle bat les baselines sur le même dev :
`logistic --relative` pour la détection, `ridge` et `constant` pour la
distance. Deux commandes, zéro ambiguïté sur le gain réel.

### La règle d'or

Le holdout (`02_evaluate.py`) se consulte **une fois par question**, après
avoir choisi sur `dev_cv`. Chaque variante supplémentaire évaluée sur le
holdout l'affaiblit. Les 9 748 fenêtres se chevauchent : ce ne sont pas
9 748 observations indépendantes.

## Sorties d'un run (`runs/<nom>/`)

- `model.joblib` — modèle + scaler + colonnes/vocabulaire, rechargeable ;
- `config.json` — params, cache utilisé, empreintes (cache + code), dev_cv ;
- `cv.csv` — score par pli de développement ;
- `curves.json` — courbes d'apprentissage par pli + réajustement final ;
- `learning-curves.png` (via 02) — train/validation par pli ;
- `metrics.json` (via 02) — holdout : overall, par téléphone, et par
  distance cible pour la régression, plus la MAE non bornée.

## Méthodologie (à ne pas casser)

Mêmes règles que la spécification ML V1 du hackathon : répartition par
**événement temporel avant fenêtrage** (un groupe ne traverse jamais
dev/test), **poids égal par prise**, scaler ajusté sur le train du pli
uniquement, fenêtres valides de 1 s / pas 250 ms après contrôle qualité,
niveaux **absolus** en dBFS pour la distance. La parité 24 kHz avec le cache
qui a servi au run RTX est testée automatiquement à chaque `pytest`.

Limites à dire telles quelles devant le jury : même campagne, mêmes trois
appareils, une seule prise à 5 m par téléphone ; ni précision 2D, ni latence
de détection, ni garantie à ±50 cm par mesure.

## Où sont passés les anciens fichiers ?

Dans l'historique Git (`18c85fc` et commits précédents) : les dossiers
`00_`–`03_` d'origine (16 kHz, fenêtres de 12,5 ms, sans split groupé), le
package `edth/` (release/publish/predict/plot) et leurs scripts. Le rapport
HTML/PDF du jury reste dans `runs/retour-rtx/`, non versionné.
