# EDTH — entraînement audio, démonstrateur 1–5 m

Deux modèles distincts sur les trois appareils connus tenus en main : probabilité de présence et distance conditionnelle à la présence. Le pipeline versionné `edth` remplace les tables exploratoires des dossiers `00`–`03` ; ne pas les exécuter en amont. Les maîtres restent des WAV PCM16 mono 48 kHz. Pas d'envoi réseau par défaut, hormis le téléchargement explicitement demandé.

## 1. Installer à la racine du dépôt

Python **3.12**, Linux x86-64. Utiliser un environnement neuf. Pour reproduire le laptop (CPU) :

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.lock
python -m pytest -q
```

Sur la **RTX 3090**, vérifier le pilote avec `nvidia-smi`. Installer une wheel CUDA de PyTorch compatible avec ce pilote (exemple CUDA 12.6, ne pas installer le lock CPU sur le serveur) :

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
python -m pytest -q
```

## 2. Préparer une révision immuable

Une seule de ces commandes, avec un dossier de sortie neuf. Le chemin local vise le dossier `data` du collecteur, **pas** ses archives d'audit. Pour le Hub, utiliser un token en lecture dans l'environnement ou dans `.env` racine (voir `.env.example`) ; ne jamais copier de secrets dans un bundle.

```sh
python -m edth.data --local /home/noe/Projects/EDTH_data-collector/data --out artifacts/pilot-v2
# OU sur le serveur : même contenu, révision vérifiée
python -m edth.data --repo ImAPancake/EDTH_drone_distance_raw --revision 811bafcd6fa0e94b287b37133d8d6331b50eb3e3 --out artifacts/pilot-v2
```

Le manifeste contient les empreintes PCM, les étiquettes, les exclusions, les groupes/folds, les coefficients FIR et les bandes. Le parquet contient les features, pas les WAV. Sur la révision pilote : 58 prises, 9 748 fenêtres, 25 groupes temporels. Le test réservé compte 15 prises ; les 43 restantes forment les trois plis de développement. **Pas 9 748 observations indépendantes.** Groupes = composantes de chevauchement temporel entre tous les appareils, tolérance 1 s ; pas des sessions garanties indépendantes. Les prises à 0,1 m sont autorisées pour la détection, exclues de la régression. Pas d'annotation automatique de transitions ou de correction destructive du brut.

Chaîne : coupures numériques ≥40 ms au 48 kHz, garde démarrage/reprise 500 ms ; FIR causal Kaiser 161 coefficients puis décimation et quantification S16. Fenêtre valide de 1 s, pas 250 ms. Énergies absolues de 32 bandes espacées sur l'échelle mel, puissance 4–8 kHz, RMS pleine bande ; aucune mise à l'échelle apprise dans le cache. La v2 conserve aussi le spectrogramme log-mel de chaque seconde (32 bandes triangulaires × 22 trames successives, FFT 2 048, pas 1 024). Il faut régénérer les features depuis les WAV : la moyenne spectrale v1 ne permet pas de reconstruire le temps. Garder les anciens runs et utiliser un nouveau répertoire d'étude. Le même `features_from_pcm24` fonctionne sur une fenêtre valide côté serveur. Un frontend live doit reproduire le contrat avec état de filtre entre paquets et masque QC ; la parité avec un navigateur réel n'est pas testée ici.

## 3. Vérifier le câblage, puis faire **une** étude finale

```sh
# Laptop : baselines + CNN et MLP classification, MLP distance, deux époques ; ne lit pas le vrai test pour entraîner/évaluer.
python scripts/smoke_cpu.py --data artifacts/pilot-v2 --out runs/smoke-01

# Serveur, à lancer dans une session persistante (p. ex. tmux).
TRIALS=24 DATA=artifacts/pilot-v2 RUN=runs/night-v2 bash scripts/train_server.sh
```

Le script serveur enchaîne les tâches sur une seule carte, journalise, trace les courbes, empaquette les deux modèles et évalue la cascade. `TRIALS` est un budget **par tâche** en plus des baselines ; 24 est un départ, pas une garantie de durée. Les petits modèles linéaires utilisent naturellement le CPU. Pour contrôler individuellement le budget :

```sh
python -m edth.train --task distance --data artifacts/pilot-v2/windows.parquet --out runs/distance-01 --device cuda --trials 24 --epochs 60 --patience 8
python -m edth.plot --run runs/distance-01
```

Optuna persiste dans SQLite par tâche ; après interruption, relancer à configuration identique, un seul processus par dossier. Un run finalisé n'est pas repris pour tuner à nouveau après consultation du test. Si les données changent, choisir de nouveaux chemins `DATA` et `RUN`, noter la nouvelle révision et réserver de nouveaux événements avant toute comparaison finale. Ne pas retenir le meilleur de plusieurs runs sur leur score de test.

**Sélection.** Même poids total par prise, folds fixés avant HPO, scaler et encodage fit sur le train du pli. Détection : logistique et MLP sur les features résumées (bandes et one-hot téléphone optionnels), ou CNN 2D sur le spectrogramme pleine bande (audio seul). Le CNN apprend des motifs locaux temps-fréquence ; Optuna teste 1 ou 2 blocs et 8/16/32 canaux initiaux. Toutes les familles de détection peuvent conserver ou retirer le niveau global ; la standardisation pour le CNN est ajustée par bande sur le train du pli uniquement. Score = balanced accuracy pondérée par prise, seuil 0,5 fixé à l'avance. Distance : 3 m constant, référence à 1 m par téléphone + modèle log-distance, Ridge ou MLP sur niveaux absolus ; score = MAE en mètres pondérée par prise. Aucune distance d'annotation, date ou durée n'entre dans les prédicteurs. Chaque réseau compare des taux de dropout de 0, 0,15 et 0,30. Avec au moins 6 essais de classification (3 pour la distance), les premiers essais comparent ces valeurs à architecture, lr et plis identiques au sein de chaque famille neuronale ; Optuna explore ensuite les autres configurations. Dropout inactif à l'inférence. Arrêt anticipé sur la loss de validation et pruning entre plis. Le gagnant est réajusté sur tout le développement ; pour un réseau, nombre d'époques choisi en CV. Si une baseline gagne, c'est elle qui est livrée.

**Lire les résultats.** Chaque tâche écrit `manifest.json`, `model.joblib` (scaler, vocabulaire, calibration éventuelle, modèle), `cv_comparison.csv` (une ligne par candidat terminé : famille, dropout, score par pli et score de sélection, **sans test**), `metrics.json` (CV, test et références), `curves.json`, `test_predictions.parquet` et `study.sqlite3`. `learning-curves.png` compare train/validation du meilleur réseau terminé (CNN ou MLP). `evaluation.json` du script serveur donne en plus fausses alertes/minute de négatif valide, couverture temporelle et MAE des distances effectivement émises par la cascade. La métrique de distance seule suppose la présence connue ; elle ne remplace pas ces chiffres de bout en bout. Pas de promesse « SOTA », de mesure du délai de détection sans annotations d'onset, ni de précision de position 2D.

## 4. Rejouer et publier en privé

```sh
python -m edth.predict --bundle runs/night-v2/bundle --wav /chemin/audio.wav --device-id UUID_APPAREIL
# Publication volontaire, uniquement vers un NOUVEAU dépôt de modèles privé ; HF_TOKEN avec écriture requis.
python -m edth.release publish --bundle runs/night-v2/bundle --repo ImAPancake/edth-audio-demo-v1
```

Le replay WAV retourne les temps des fenêtres **valides** et, pour chacune, une probabilité et une distance en mètres seulement si le seuil est atteint. Un intervalle manquant signifie indisponible, pas absent. Domaine des sorties de distance : 1–5 m, sans certification que la source est réellement dans cette plage. Pas de streaming, hystérésis ou trilatération dans ce repo. Le bundle contient les sources d'inférence, deux modèles, versions, paramètres et empreintes ; ni token, ni audio, ni base Optuna. Télécharger une révision précise de ce dépôt privé puis lancer la même commande depuis les sources livrées. Ne charger des pickles/joblib que d'une provenance de confiance ; les empreintes détectent les incohérences, pas un éditeur malveillant.
