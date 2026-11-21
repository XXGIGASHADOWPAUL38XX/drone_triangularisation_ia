"""Tune the MLP on a fixed feature cache for each task."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "01_ppc_features")]

import ml
from dsp import DEFAULT_FEATURES, default_window_cache, resolve_features

SETTINGS = {
    "classification": ("02_ppc_classification", 12, 4),
    "distance": ("03_ppc_distance", 60, 8),
}
BASELINE = dict(lr=0.003, dropout=0.15, hidden_units=16, batch_size=128, weight_decay=0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=SETTINGS)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--name", default="optuna")
    parser.add_argument("--input", type=Path, help="Override the task's fixed feature cache")
    parser.add_argument("--objective-split", choices=["validation", "holdout"], default="validation")
    parser.add_argument("--max-depth", type=int, choices=[1, 2, 3], default=1,
                        help="Search from one hidden layer up to this depth")
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("--trials and --threads must be positive")
    torch.set_num_threads(args.threads)

    stage, epochs, patience = SETTINGS[args.task]
    cache = args.input or default_window_cache(ROOT / stage / "input", classification=args.task == "classification")
    out = ROOT / "04_compare" / "output" / args.name / args.task
    out.mkdir(parents=True, exist_ok=False)
    frame = ml.task_frame(ml.load(cache), args.task)
    train, valid, test = (frame[frame.split == split] for split in ("train", "validation", "test"))
    if train.empty or valid.empty or test.empty:
        parser.error("the cache must provide nonempty train, validation, and holdout splits")
    objective_frame = valid if args.objective_split == "validation" else test
    cols = resolve_features(DEFAULT_FEATURES)
    metric = ml.selection_metric(args.task)
    seed = 42
    study = optuna.create_study(
        direction="maximize" if args.task == "classification" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.enqueue_trial(dict(BASELINE, depth=1) if args.max_depth > 1 else BASELINE)
    best_model = None

    def objective(trial):
        params = dict(
            lr=trial.suggest_float("lr", 0.0003, 0.01, log=True),
            dropout=trial.suggest_float("dropout", 0, 0.30, step=0.05),
            hidden_units=trial.suggest_categorical("hidden_units", [8, 16, 32, 64]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            weight_decay=trial.suggest_float("weight_decay", 0.00001, 0.01, log=True),
        )
        if args.max_depth > 1:
            params["depth"] = trial.suggest_int("depth", 1, args.max_depth)
        model = ml.Mlp(args.task, **params)
        model.fit(train, valid, epochs=epochs, patience=patience, seed=seed, cols=cols)
        trial.set_user_attr("best_epoch", model.best_epoch)
        validation = ml.metrics(valid.assign(pred=model.predict(valid)), args.task)[metric]
        trial.set_user_attr("validation", validation)
        score = validation if objective_frame is valid else ml.metrics(
            test.assign(pred=model.predict(test)), args.task
        )[metric]
        return score, model

    for _ in range(args.trials):
        trial = study.ask()
        score, model = objective(trial)
        study.tell(trial, score)
        if study.best_trial.number == trial.number:
            best_model = model
        print(f"{args.task} trial {trial.number + 1}/{args.trials}  "
              f"{args.objective_split} {metric}={score:.6f}  best={study.best_value:.6f}", flush=True)
        study.trials_dataframe().to_csv(out / "trials.csv", index=False)

    holdout = ml.report(test, best_model.predict(test), args.task)
    summary = dict(
        task=args.task, cache=cache.name, features=cols, seed=seed, epochs=epochs,
        patience=patience, trials=args.trials, metric=metric, max_depth=args.max_depth,
        objective_split=args.objective_split, baseline_objective=study.trials[0].value,
        best_objective=study.best_value, baseline_validation=study.trials[0].user_attrs["validation"],
        best_trial=study.best_trial.number, best_params=study.best_params,
        best_epoch=best_model.best_epoch, validation=study.best_trial.user_attrs["validation"], holdout=holdout,
    )
    joblib.dump(best_model, out / "model.joblib")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{args.task} holdout {metric}={holdout['overall'][metric]:.6f}  {out}", flush=True)


if __name__ == "__main__":
    main()
