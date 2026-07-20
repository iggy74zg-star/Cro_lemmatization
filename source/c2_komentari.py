#!/usr/bin/env python3
"""
End-to-end pipeline:
- TF-IDF experiments: RandomForest, SVM (RBF), LinearSVM, NeuralNet (original and lemmatized)
- Transformer experiments: croBERT-style (fine-tune via Hugging Face Trainer) for original and lemmatized text
- Saves models, confusion matrices, ROC curves (markers every 10th point)
- Aggregates all metrics into CSV + JSON: OUTPUT_DIR/all_metrics_with_crobert.csv/.json
- Records duration per step (seconds) and total runtime
- OUTPUT_DIR set to /Users/iggy/models_out
"""

import argparse
import os
import re
import time
import json
import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from scipy import sparse
from joblib import dump

import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# stanza for Croatian lemmatization
import stanza

# imbalanced-learn SMOTE
from imblearn.over_sampling import SMOTE

# transformers / datasets for croBERT fine-tuning
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# ---------------------------
# Config
# ---------------------------
ARROW_PATH = os.environ.get("C2_ARROW_PATH", "/Users/iggy/Downloads/datasets/enriched_frenk_2306_ds/data-00000-of-00001.arrow")
LABEL_COL = "label"
RANDOM_STATE = 42
MODEL_NAME = os.environ.get("C2_MODEL_NAME", "bert-base-multilingual-cased")  # replace with croBERT checkpoint if available

# OUTPUT_DIR as requested
OUTPUT_DIR = os.environ.get("C2_OUTPUT_DIR", "/Users/iggy/models_out")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SERIES_CONFIG = {
    "original": {"MAX_TFIDF_FEATURES": 10000, "WORD_NGRAMS": (1,2), "CHAR_NGRAMS": None, "SVD_COMPONENTS": 200, "N_ITER_SEARCH": 8},
    "lemmatized": {"MAX_TFIDF_FEATURES": 20000, "WORD_NGRAMS": (1,2), "CHAR_NGRAMS": (3,5), "SVD_COMPONENTS": 400, "N_ITER_SEARCH": 16}
}

TRAIN_RATIO = 0.15
N_JOBS = -1
HEX_RE = re.compile(r'^[0-9a-fA-F]+$')


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end TF-IDF and transformer experiments for the Croatian text classification pipeline.")
    parser.add_argument("--data-path", default=ARROW_PATH, help="Path to the input Arrow dataset file.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory used to store models, metrics, and plots.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name for the transformer experiments.")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="Train split ratio used for fallback single-split runs.")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="Number of cross-validation folds.")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="Number of random seeds used for repeated CV runs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation to reduce runtime and output size.")
    return parser.parse_args()

TRAINING_ARGS = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01
}
# Cross-validation config
CV_FOLDS = 5
# Increase NUM_SEEDS so CV_FOLDS * NUM_SEEDS = total resampling runs (e.g., 5 * 7 = 35)
NUM_SEEDS = 7


# ---------------------------
# Utilities
# ---------------------------
def read_arrow_to_df(path):
    with open(path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    return table.to_pandas()

def looks_like_hex_series(s: pd.Series, min_ratio=0.9, min_len=4) -> bool:
    if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
        return False
    sample = s.dropna().astype(str)
    if sample.empty:
        return False
    is_hex = sample.apply(lambda x: bool(HEX_RE.fullmatch(x)) and len(x) >= min_len)
    return is_hex.mean() >= min_ratio

def hex_decode_series(s: pd.Series) -> pd.Series:
    def try_decode(x):
        try:
            b = bytes.fromhex(x)
            return b.decode("utf-8", errors="strict")
        except Exception:
            try:
                return bytes.fromhex(x).decode("latin1", errors="ignore")
            except Exception:
                return x
    return s.fillna("").astype(str).apply(try_decode)

# ---------------------------
# Stanza lemmatization
# ---------------------------
def ensure_stanza_hr():
    try:
        stanza.download('hr', verbose=False)
    except Exception:
        pass
    return stanza.Pipeline(lang='hr', processors='tokenize,pos,lemma', verbose=False)

def lemmatize_series_stanza(text_series: pd.Series, nlp) -> pd.Series:
    out = []
    for t in text_series.fillna("").astype(str):
        if not t:
            out.append("")
            continue
        doc = nlp(t)
        tokens = [w.lemma if getattr(w, "lemma", None) else w.text for s in doc.sentences for w in s.words]
        out.append(" ".join(tokens))
    return pd.Series(out, index=text_series.index)

# ---------------------------
# Feature prep (TF-IDF) returning also raw text_series
# ---------------------------
def prepare_features(df: pd.DataFrame, series_key: str, apply_lemmatization=False, nlp=None):
    cfg = SERIES_CONFIG[series_key]
    df = df.copy()
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found")
    for c in df.columns:
        if c == LABEL_COL:
            continue
        try:
            if looks_like_hex_series(df[c]):
                df[c] = hex_decode_series(df[c])
        except Exception:
            pass

    y_raw = df[LABEL_COL].values
    X_df = df.drop(columns=[LABEL_COL])

    text_cols = [c for c in X_df.columns if pd.api.types.is_object_dtype(X_df[c]) or pd.api.types.is_string_dtype(X_df[c])]
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]

    text_series = None
    if text_cols:
        def concat_text(row):
            parts = []
            for c in text_cols:
                v = row.get(c)
                if pd.isna(v):
                    continue
                parts.append(str(v))
            return " ".join(parts)
        text_series = X_df[text_cols].apply(lambda r: concat_text(r), axis=1)
        if apply_lemmatization:
            if nlp is None:
                raise ValueError("Stanza pipeline required for lemmatization")
            print("Lemmatizing text (this can be slow)...")
            text_series = lemmatize_series_stanza(text_series, nlp)

        if cfg["CHAR_NGRAMS"] is None:
            tfidf = TfidfVectorizer(max_features=cfg["MAX_TFIDF_FEATURES"], ngram_range=cfg["WORD_NGRAMS"], analyzer="word")
            X_text = tfidf.fit_transform(text_series)
        else:
            tfw = TfidfVectorizer(max_features=int(cfg["MAX_TFIDF_FEATURES"]*0.7), ngram_range=cfg["WORD_NGRAMS"], analyzer="word")
            tfc = TfidfVectorizer(max_features=int(cfg["MAX_TFIDF_FEATURES"]*0.3), ngram_range=cfg["CHAR_NGRAMS"], analyzer="char_wb")
            Xw = tfw.fit_transform(text_series)
            Xc = tfc.fit_transform(text_series)
            from scipy.sparse import hstack
            X_text = hstack([Xw, Xc], format="csr")
            tfidf = (tfw, tfc)
    else:
        tfidf = None
        X_text = None

    if numeric_cols:
        X_num_df = X_df[numeric_cols].copy()
        imp = SimpleImputer(strategy="median")
        X_num = imp.fit_transform(X_num_df)
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = None

    if X_text is not None and X_num is not None:
        from scipy.sparse import hstack, csr_matrix
        X_final = hstack([X_text, csr_matrix(X_num)], format="csr")
    elif X_text is not None:
        X_final = X_text
    elif X_num is not None:
        X_final = X_num
    else:
        raise ValueError("No usable features")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    meta = {"tfidf": tfidf, "label_encoder": le, "lemmatized": apply_lemmatization, "series_cfg": cfg}
    return X_final, y, meta, text_series

# ---------------------------
# Representation and plotting helpers
# ---------------------------
from scipy.special import softmax

def ensure_same_representation(X_train, X_test, use_svd=False, svd_components=200):
    if sparse.issparse(X_train) and use_svd:
        svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)
        Xtr = svd.fit_transform(X_train)
        Xte = svd.transform(X_test)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
        return Xtr, Xte, ("SVD+Scaler", svd, scaler)
    if sparse.issparse(X_train) and sparse.issparse(X_test):
        return X_train, X_test, None
    if sparse.issparse(X_train) and not sparse.issparse(X_test):
        return X_train, sparse.csr_matrix(X_test), None
    if not sparse.issparse(X_train) and sparse.issparse(X_test):
        return sparse.csr_matrix(X_train), X_test, None
    return X_train, X_test, None

def compute_metrics(y_true, y_pred, model=None, X_test=None):
    res = {}
    res["accuracy"] = float(accuracy_score(y_true, y_pred))
    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
    res["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    res["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    res["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    res["roc_auc"] = None
    res["roc_auc_macro"] = None
    try:
        if model is not None and X_test is not None:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                if probs is not None:
                    if probs.shape[1] == 2:
                        res["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
                    else:
                        res["roc_auc_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovo", average="macro"))
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(X_test)
                if dec is not None and dec.ndim == 1:
                    res["roc_auc"] = float(roc_auc_score(y_true, dec))
    except Exception:
        pass
    return res

def plot_confusion_matrix(cm, labels, out_path, title="Confusion matrix"):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def _plot_roc_line_with_markers(x, y, label, color, lw=2, marker_step=10):
    plt.plot(x, y, color=color, lw=lw, label=label)
    if len(x) >= marker_step:
        idxs = np.arange(0, len(x), marker_step)
        plt.plot(x[idxs], y[idxs], linestyle='None', marker='o', markersize=4, color=color, alpha=0.85)

from sklearn.preprocessing import label_binarize
def plot_roc_curve_multiclass(model, X_test, y_test, label_encoder, out_path, title="ROC Curve", marker_step=10):
    classes = label_encoder.classes_
    n_classes = len(classes)
    try:
        y_test_b = label_binarize(y_test, classes=np.arange(n_classes))
    except Exception:
        y_unique = np.unique(y_test)
        y_test_b = label_binarize(y_test, classes=y_unique)
        classes = y_unique; n_classes = len(classes)
    plt.figure(figsize=(8,6)); colors = plt.cm.get_cmap('tab10')
    if n_classes == 2:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)
            if probs.ndim > 1:
                probs = softmax(probs, axis=1)[:, 1]
        else:
            plt.text(0.5, 0.5, "No probability/decision function available", horizontalalignment='center'); plt.savefig(out_path, dpi=150); plt.close(); return
        fpr, tpr, _ = roc_curve(y_test, probs); roc_auc = auc(fpr, tpr)
        _plot_roc_line_with_markers(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='darkorange', lw=2, marker_step=marker_step)
        plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(); return
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X_test)
        try:
            probs = softmax(probs, axis=1)
        except Exception:
            pass
    else:
        plt.text(0.5, 0.5, "No prob/decision function available for multiclass ROC", horizontalalignment='center'); plt.savefig(out_path, dpi=150); plt.close(); return
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], probs[:, i]); roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i], tpr[i], roc_auc[i] = None, None, None
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_b.ravel(), probs.ravel()); roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        roc_auc["micro"] = None
    for i in range(n_classes):
        if fpr[i] is None: continue
        _plot_roc_line_with_markers(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})', color=colors(i % 10), lw=1.5, marker_step=marker_step)
    if roc_auc.get("micro") is not None and fpr.get("micro") is not None:
        _plot_roc_line_with_markers(fpr["micro"], tpr["micro"], label=f'micro-average (AUC = {roc_auc["micro"]:.3f})', color='black', lw=2.0, marker_step=marker_step*2)
    plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--'); plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend(loc='lower right', fontsize='small'); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _get_probs_from_model(model, X_test):
    """Return predicted probabilities or decision outputs suitable for ROC calculation.
    May return None if no probabilities available."""
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            dec = model.decision_function(X_test)
            if np.ndim(dec) == 1:
                probs = dec
            else:
                try:
                    probs = softmax(dec, axis=1)
                except Exception:
                    probs = dec
    except Exception:
        probs = None
    return probs


def compute_roc_data_for_model(model, X_test, y_test, label_encoder):
    """Compute ROC curve data dict for binary or multiclass model.
    Returns dict with keys: 'binary' or 'per_class' and 'micro' as available."""
    classes = label_encoder.classes_
    n_classes = len(classes)
    probs = _get_probs_from_model(model, X_test)
    roc_data = {}
    if n_classes == 2:
        if probs is None:
            return None
        if probs.ndim > 1:
            score = probs[:, 1]
        else:
            score = probs
        fpr, tpr, _ = roc_curve(y_test, score)
        roc_data['binary'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        return roc_data

    # multiclass
    try:
        y_test_b = label_binarize(y_test, classes=np.arange(n_classes))
    except Exception:
        y_unique = np.unique(y_test)
        y_test_b = label_binarize(y_test, classes=y_unique)
        n_classes = y_test_b.shape[1]
    if probs is None:
        return None
    per_class = {}
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_test_b[:, i], probs[:, i])
            per_class[i] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
        except Exception:
            per_class[i] = None
    roc_data['per_class'] = per_class
    try:
        fpr_m, tpr_m, _ = roc_curve(y_test_b.ravel(), probs.ravel())
        roc_data['micro'] = {'fpr': fpr_m, 'tpr': tpr_m, 'auc': auc(fpr_m, tpr_m)}
    except Exception:
        roc_data['micro'] = None
    return roc_data

# ---------------------------
# Classic models runner
# ---------------------------
def run_classic_models(X, y, meta, series_key, save_plots_for_series=False, all_metrics_records=None, cv_splits_dict=None, seeds=None):
    cfg = SERIES_CONFIG[series_key]
    print(f"\nClassic models: {series_key}")
    results = {}
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
    except TypeError:
        smote = SMOTE(random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf_params = {"max_depth":[None,20,40],"max_features":["sqrt",0.2,0.5],"min_samples_leaf":[1,2,4]}
    svm_rbf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    svm_params = {"C":[0.5,1,5],"gamma":["scale",0.01,0.001]}
    linear = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)
    linear_params = {"C":[0.01,0.1,1,10]}
    nn = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=RANDOM_STATE)
    nn_params = {"alpha":[1e-5,1e-4,1e-3],"learning_rate_init":[1e-4,1e-3]}

    def train_eval(name, model, params=None, require_dense=False, Xtr=None, Xte=None, ytr=None, yte=None, seed=None, fold=None):
        t0 = time.time()
        print(f"Training {name} (seed={seed} fold={fold})...")
        Xtr_r, Xte_r, tinfo = ensure_same_representation(Xtr, Xte, use_svd=require_dense, svd_components=cfg["SVD_COMPONENTS"])
        Xfit, yfit = Xtr_r, ytr
        if not sparse.issparse(Xfit):
            try:
                Xfit, yfit = smote.fit_resample(Xfit, yfit)
                print("SMOTE applied:", np.bincount(yfit))
            except Exception as e:
                print("SMOTE warn:", e)
        best = model
        if params:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scoring = "f1_macro" if len(np.unique(yfit))>2 else "f1"
            search = RandomizedSearchCV(model, param_distributions=params, n_iter=8, scoring=scoring, n_jobs=N_JOBS, cv=cv, random_state=RANDOM_STATE, verbose=0)
            search.fit(Xfit, yfit); best = search.best_estimator_; print("best params:", search.best_params_)
        else:
            best.fit(Xfit, yfit)
        ypred = best.predict(Xte_r)
        metrics = compute_metrics(yte, ypred, model=best, X_test=Xte_r)
        cm = confusion_matrix(yte, ypred)
        run_id = f"s{seed}_f{fold}"
        save_name = f"{series_key}_{name}_{run_id}"
        dump({"model":best,"meta":meta,"transform_info":tinfo}, os.path.join(OUTPUT_DIR, f"{save_name}.joblib"))
        if save_plots_for_series:
            labels = list(meta["label_encoder"].classes_)
            plot_confusion_matrix(cm, labels, os.path.join(PLOTS_DIR, f"{save_name}_confusion.png"), title=f"{series_key}/{name} CM {run_id}")
            try:
                plot_roc_curve_multiclass(best, Xte_r, yte, meta["label_encoder"], os.path.join(PLOTS_DIR, f"{save_name}_roc.png"), title=f"{series_key}/{name} ROC {run_id}", marker_step=10)
            except Exception as e:
                print("ROC plot warn:", e)
        duration = time.time() - t0
        metrics_record = {"series": series_key, "model": name, "seed": int(seed), "fold": int(fold), "run_id": run_id, **metrics, "duration_sec": float(duration)}
        if all_metrics_records is not None:
            all_metrics_records.append(metrics_record)
        print(f"{name} done in {duration:.1f}s; f1={metrics.get('f1', float('nan')):.4f}")
        return {"metrics": metrics, "cm": cm, "model": best, "duration_sec": duration, "X_test": Xte_r, "y_test": yte, "tinfo": tinfo, "run_id": run_id}
    # If cv_splits_dict and seeds provided, run CV x seeds
    if cv_splits_dict and seeds:
        for seed in seeds:
            splits = cv_splits_dict.get(seed)
            if splits is None:
                continue
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                ytr = y[train_idx]; yte = y[test_idx]
                Xtr = X[train_idx]; Xte = X[test_idx]
                # run each model per fold
                results.setdefault('RandomForest', [])
                results.setdefault('SVM', [])
                results.setdefault('LinearSVM', [])
                results.setdefault('NeuralNet', [])
                results['RandomForest'].append(train_eval("RandomForest", rf, params=rf_params, require_dense=False, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, seed=seed, fold=fold_idx))
                results['SVM'].append(train_eval("SVM", svm_rbf, params=svm_params, require_dense=True, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, seed=seed, fold=fold_idx))
                results['LinearSVM'].append(train_eval("LinearSVM", linear, params=linear_params, require_dense=False, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, seed=seed, fold=fold_idx))
                results['NeuralNet'].append(train_eval("NeuralNet", nn, params=nn_params, require_dense=True, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, seed=seed, fold=fold_idx))
        return results

    # Fallback: original single split behavior
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_RATIO, test_size=1-TRAIN_RATIO, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None)
    results["RandomForest"] = train_eval("RandomForest", rf, params=rf_params, require_dense=False, Xtr=X_train, Xte=X_test, ytr=y_train, yte=y_test, seed=RANDOM_STATE, fold=0)
    results["SVM"] = train_eval("SVM", svm_rbf, params=svm_params, require_dense=True, Xtr=X_train, Xte=X_test, ytr=y_train, yte=y_test, seed=RANDOM_STATE, fold=0)
    results["LinearSVM"] = train_eval("LinearSVM", linear, params=linear_params, require_dense=False, Xtr=X_train, Xte=X_test, ytr=y_train, yte=y_test, seed=RANDOM_STATE, fold=0)
    results["NeuralNet"] = train_eval("NeuralNet", nn, params=nn_params, require_dense=True, Xtr=X_train, Xte=X_test, ytr=y_train, yte=y_test, seed=RANDOM_STATE, fold=0)
    return results

# ---------------------------
# Transformer runner (fine-tune) — returns metrics with same keys and duration
# ---------------------------
def run_transformer(text_series: pd.Series, labels, label_encoder: LabelEncoder, series_key: str, save_plots=True, all_metrics_records=None, cv_splits_dict=None, seeds=None):
    # backward compatible: this function can also accept cv_splits_dict and seeds via kwargs in future
    t0 = time.time()
    print(f"\nTransformer training for {series_key} (this may take long)...")
    num_labels = len(label_encoder.classes_)
    runs = []
    # If cv_splits_dict and seeds provided, perform CV x seeds for transformer fine-tuning
    if cv_splits_dict and seeds:
        for seed in seeds:
            splits = cv_splits_dict.get(seed)
            if splits is None:
                continue
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train_text = text_series.values[train_idx]
                X_test_text = text_series.values[test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
                ds_train = Dataset.from_dict({"text": X_train_text.tolist(), "label": y_train.tolist()})
                ds_test = Dataset.from_dict({"text": X_test_text.tolist(), "label": y_test.tolist()})
                ds_run = DatasetDict({"train": ds_train, "test": ds_test})
                ds_run = ds_run.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256), batched=True)
                ds_run.set_format(type="torch", columns=["input_ids","attention_mask","label"])
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
                run_output_dir = os.path.join(OUTPUT_DIR, f"{series_key}_transformer_s{seed}_f{fold_idx}")
                training_args = TrainingArguments(output_dir=run_output_dir, num_train_epochs=TRAINING_ARGS["num_train_epochs"], per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"], per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"], learning_rate=TRAINING_ARGS["learning_rate"], weight_decay=TRAINING_ARGS["weight_decay"], seed=int(seed), logging_steps=50, save_strategy="no", evaluation_strategy="no")
                def compute_metrics_hf(p):
                    logits = p.predictions
                    if isinstance(logits, tuple): logits = logits[0]
                    ypred = np.argmax(logits, axis=1)
                    avg = "binary" if num_labels==2 else "macro"
                    return {"accuracy": float((ypred==p.label_ids).mean()), "f1": float(f1_score(p.label_ids, ypred, average=avg, zero_division=0))}
                trainer = Trainer(model=model, args=training_args, train_dataset=ds_run["train"], eval_dataset=ds_run["test"], compute_metrics=compute_metrics_hf)
                print(f"Training transformer {series_key} seed={seed} fold={fold_idx}...")
                trainer.train()
                pred_out = trainer.predict(ds_run["test"])
                logits = pred_out.predictions
                if isinstance(logits, tuple): logits = logits[0]
                y_pred = np.argmax(logits, axis=1)
                metrics = {}
                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
                avg = "binary" if num_labels==2 else "macro"
                metrics["precision"] = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
                metrics["recall"] = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
                metrics["f1"] = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
                metrics["roc_auc"] = None
                metrics["roc_auc_macro"] = None
                probs_arr = None
                try:
                    if logits is not None:
                        probs_arr = torch.softmax(torch.tensor(logits), dim=1).numpy() if logits.ndim>1 else None
                        if probs_arr is not None:
                            if probs_arr.shape[1] == 2:
                                metrics["roc_auc"] = float(roc_auc_score(y_test, probs_arr[:,1]))
                            else:
                                metrics["roc_auc_macro"] = float(roc_auc_score(y_test, probs_arr, multi_class="ovo", average="macro"))
                except Exception:
                    pass
                # save per-run model/tokenizer
                try:
                    model.save_pretrained(run_output_dir); tokenizer.save_pretrained(run_output_dir)
                except Exception:
                    pass
                duration_run = time.time() - t0
                run_id = f"s{seed}_f{fold_idx}"
                if save_plots:
                    labels = list(label_encoder.classes_)
                    cm = confusion_matrix(y_test, y_pred)
                    plot_confusion_matrix(cm, labels, os.path.join(PLOTS_DIR, f"{series_key}_Transformer_confusion_{run_id}.png"), title=f"{series_key} Transformer CM {run_id}")
                    if probs_arr is not None:
                        class ProbWrapper:
                            def __init__(self, probs): self.probs = probs
                            def predict_proba(self, X): return self.probs
                        wrapper = ProbWrapper(probs_arr)
                        try:
                            plot_roc_curve_multiclass(wrapper, None, y_test, label_encoder, os.path.join(PLOTS_DIR, f"{series_key}_Transformer_roc_{run_id}.png"), title=f"{series_key} Transformer ROC {run_id}", marker_step=10)
                        except Exception as e:
                            print("Transformer ROC warn:", e)
                metrics_record = {"series": series_key, "model": "croBERT", "seed": int(seed), "fold": int(fold_idx), "run_id": run_id, **metrics, "duration_sec": float(duration_run)}
                if all_metrics_records is not None:
                    all_metrics_records.append(metrics_record)
                runs.append({"metrics": metrics, "save_dir": run_output_dir, "y_test": y_test, "y_pred": y_pred, "probs_arr": probs_arr, "duration_sec": duration_run, "seed": int(seed), "fold": int(fold_idx), "run_id": run_id})
        # aggregate metrics across runs
        if runs:
            agg = {}
            for m in ["f1", "accuracy", "precision", "recall"]:
                vals = [float(r["metrics"].get(m, float('nan')) or float('nan')) for r in runs]
                vals = [x for x in vals if not (x is None or (isinstance(x, float) and np.isnan(x)))]
                agg[m] = float(np.mean(vals)) if vals else float('nan')
            return {"metrics": agg, "runs": runs}
        return {"metrics": {}, "runs": []}

    # Fallback single-run behavior (as before)
    X_train_text, X_test_text, y_train, y_test = train_test_split(text_series.values, labels, train_size=TRAIN_RATIO, test_size=1-TRAIN_RATIO, random_state=RANDOM_STATE, stratify=labels if len(np.unique(labels))>1 else None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds_train = Dataset.from_dict({"text": X_train_text.tolist(), "label": y_train.tolist()})
    ds_test = Dataset.from_dict({"text": X_test_text.tolist(), "label": y_test.tolist()})
    ds = DatasetDict({"train": ds_train, "test": ds_test})
    ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=256), batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    output_dir = os.path.join(OUTPUT_DIR, f"{series_key}_transformer")
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=TRAINING_ARGS["num_train_epochs"], per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"], per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"], learning_rate=TRAINING_ARGS["learning_rate"], weight_decay=TRAINING_ARGS["weight_decay"], seed=RANDOM_STATE, logging_steps=50, save_strategy="no", evaluation_strategy="no")
    def compute_metrics_hf(p):
        logits = p.predictions
        if isinstance(logits, tuple): logits = logits[0]
        ypred = np.argmax(logits, axis=1)
        avg = "binary" if num_labels==2 else "macro"
        return {"accuracy": float((ypred==p.label_ids).mean()), "f1": float(f1_score(p.label_ids, ypred, average=avg, zero_division=0))}
    trainer = Trainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["test"], compute_metrics=compute_metrics_hf)
    trainer.train()
    pred_out = trainer.predict(ds["test"])
    logits = pred_out.predictions
    if isinstance(logits, tuple): logits = logits[0]
    y_pred = np.argmax(logits, axis=1)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    avg = "binary" if num_labels==2 else "macro"
    metrics["precision"] = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
    metrics["roc_auc"] = None
    metrics["roc_auc_macro"] = None
    probs_arr = None
    try:
        if logits is not None:
            probs_arr = torch.softmax(torch.tensor(logits), dim=1).numpy() if logits.ndim>1 else None
            if probs_arr is not None:
                if probs_arr.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs_arr[:,1]))
                else:
                    metrics["roc_auc_macro"] = float(roc_auc_score(y_test, probs_arr, multi_class="ovo", average="macro"))
    except Exception:
        pass
    try:
        model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    except Exception:
        pass
    duration = time.time() - t0
    if save_plots:
        labels = list(label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels, os.path.join(PLOTS_DIR, f"{series_key}_Transformer_confusion.png"), title=f"{series_key} Transformer CM")
        if probs_arr is not None:
            class ProbWrapper:
                def __init__(self, probs): self.probs = probs
                def predict_proba(self, X): return self.probs
            wrapper = ProbWrapper(probs_arr)
            try:
                plot_roc_curve_multiclass(wrapper, None, y_test, label_encoder, os.path.join(PLOTS_DIR, f"{series_key}_Transformer_roc.png"), title=f"{series_key} Transformer ROC", marker_step=10)
            except Exception as e:
                print("Transformer ROC warn:", e)
    metrics_record = {"series": series_key, "model": "croBERT", **metrics, "duration_sec": float(duration)}
    if all_metrics_records is not None:
        all_metrics_records.append(metrics_record)
    print(f"Transformer {series_key} done in {duration:.1f}s; f1={metrics['f1']:.4f}")
    return {"metrics": metrics, "save_dir": output_dir, "y_test": y_test, "y_pred": y_pred, "probs_arr": probs_arr, "duration_sec": duration}

# ---------------------------
# Save aggregated metrics helper
# ---------------------------
def save_aggregated_metrics(records, out_dir):
    df = pd.DataFrame(records)
    expected_cols = ["series", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_macro", "duration_sec"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    df = df[expected_cols]
    csv_path = os.path.join(out_dir, "all_metrics_with_crobert.csv")
    json_path = os.path.join(out_dir, "all_metrics_with_crobert.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(df.to_dict(orient="records"), jf, ensure_ascii=False, indent=2)
    print(f"Saved aggregated metrics CSV to {csv_path}")
    print(f"Saved aggregated metrics JSON to {json_path}")
    return df

# ---------------------------
# Main flow
# ---------------------------
def main():
    args = parse_args()
    global OUTPUT_DIR, PLOTS_DIR, ARROW_PATH, MODEL_NAME, TRAIN_RATIO, CV_FOLDS, NUM_SEEDS

    ARROW_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    MODEL_NAME = args.model_name
    TRAIN_RATIO = args.train_ratio
    CV_FOLDS = args.cv_folds
    NUM_SEEDS = args.num_seeds
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    total_start = time.time()
    print("Loading dataset...")
    print(f"Using data path: {ARROW_PATH}")
    print(f"Using output dir: {OUTPUT_DIR}")
    df = read_arrow_to_df(ARROW_PATH)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    nlp = ensure_stanza_hr()

    # prepare global splits (same for original and lemmatized)
    labels_global = df[LABEL_COL].values
    le_global = LabelEncoder()
    labels_enc = le_global.fit_transform(labels_global)
    seeds = [RANDOM_STATE + i for i in range(NUM_SEEDS)]
    cv_splits_dict = {}
    for s in seeds:
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=int(s))
        cv_splits_dict[s] = list(skf.split(np.arange(len(df)), labels_enc))

    all_metrics_records = []
    comparison = {"original": {}, "lemmatized": {}}
    # store full results (models, test splits, meta) for ROC aggregation
    results_store = {}

    for series_key, do_lem in [("original", False), ("lemmatized", True)]:
        series_start = time.time()
        print(f"\n=== Series {series_key} (lemmatized={do_lem}) ===")
        X, y, meta, text_series = prepare_features(df, series_key, apply_lemmatization=do_lem, nlp=nlp if do_lem else None)
        results_store[series_key] = {"meta": meta}

        # classic models
        # Save plots for both original and lemmatized series
        classic = run_classic_models(X, y, meta, series_key, save_plots_for_series=not args.no_plots, all_metrics_records=all_metrics_records, cv_splits_dict=cv_splits_dict, seeds=seeds)
        results_store[series_key]["classic"] = classic
        # aggregate per-fold/seed metrics for comparison summary
        for k, v in classic.items():
            if isinstance(v, list):
                metrics_list = [entry["metrics"] for entry in v if entry and entry.get("metrics")]
                if not metrics_list:
                    comparison[series_key][k] = {}
                    continue
                agg = {}
                for m in ["f1", "accuracy", "precision", "recall"]:
                    vals = [float(met.get(m, float('nan')) or float('nan')) for met in metrics_list]
                    vals = [x for x in vals if not (x is None or (isinstance(x, float) and np.isnan(x)))]
                    agg[m] = float(np.mean(vals)) if vals else float('nan')
                comparison[series_key][k] = agg
            else:
                comparison[series_key][k] = v.get("metrics", {})

        # transformer
        if text_series is None:
            print("No text for transformer; skipping")
        else:
            # Save transformer plots for both original and lemmatized series
            trans = run_transformer(text_series, y, meta["label_encoder"], series_key, save_plots=not args.no_plots, all_metrics_records=all_metrics_records, cv_splits_dict=cv_splits_dict, seeds=seeds)
            comparison[series_key]["croBERT"] = trans["metrics"]
            results_store[series_key]["transformer"] = trans

        series_duration = time.time() - series_start
        print(f"Series {series_key} completed in {series_duration:.1f}s")

    # Save aggregated metrics
    df_all = save_aggregated_metrics(all_metrics_records, OUTPUT_DIR)

    if not args.no_plots:
        # Comparison F1 bar chart (include croBERT)
        model_order = ["RandomForest", "SVM", "LinearSVM", "NeuralNet", "croBERT"]
        f1_orig = [comparison.get('original', {}).get(m, {}).get('f1', np.nan) for m in model_order]
        f1_lem = [comparison.get('lemmatized', {}).get(m, {}).get('f1', np.nan) for m in model_order]
        x = np.arange(len(model_order)); width = 0.35
        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar(x - width/2, f1_orig, width, label="original")
        ax.bar(x + width/2, f1_lem, width, label="lemmatized")
        ax.set_xticks(x); ax.set_xticklabels(model_order, rotation=15)
        ax.set_ylabel("F1 score"); ax.set_title("F1 score comparison (classic vs croBERT)"); ax.legend(loc='lower right')
        plt.tight_layout()
        comp_path = os.path.join(PLOTS_DIR, "comparison_f1_crobert.png")
        plt.savefig(comp_path, dpi=150); plt.close()
        print("Saved comparison plot to", comp_path)

        # --- ROC comparison plots per model (original vs lemmatized) ---
        models_to_plot = ["RandomForest", "SVM", "LinearSVM", "NeuralNet", "croBERT"]
        for model_name in models_to_plot:
            plt.figure(figsize=(8,6))
            ax = plt.gca()
            any_plotted = False
            for series_key in ["original", "lemmatized"]:
                rs = results_store.get(series_key, {})
                if not rs:
                    continue
                label_encoder = rs.get("meta", {}).get("label_encoder")
                if model_name == "croBERT":
                    trans = rs.get("transformer")
                    if not trans:
                        continue
                    # support multi-run transformer (CV x seeds) -> pick last run as representative for ROC plotting
                    if isinstance(trans, dict) and trans.get("runs"):
                        last_run = trans["runs"][-1]
                        probs = last_run.get("probs_arr")
                        y_test = last_run.get("y_test")
                    else:
                        probs = trans.get("probs_arr")
                        y_test = trans.get("y_test")
                    if probs is None or y_test is None:
                        continue
                    class ProbWrap:
                        def __init__(self, probs): self.probs = probs
                        def predict_proba(self, X): return self.probs
                    model_like = ProbWrap(probs)
                    Xtest_for_model = None
                else:
                    classic = rs.get("classic") or {}
                    info = classic.get(model_name)
                    if not info:
                        continue
                    # info may be a list of per-fold results; pick the last trained model as representative
                    if isinstance(info, list):
                        info_item = info[-1]
                    else:
                        info_item = info
                    model_like = info_item.get("model")
                    Xtest_for_model = info_item.get("X_test")
                    y_test = info_item.get("y_test")
                if label_encoder is None or y_test is None:
                    continue
                roc_data = compute_roc_data_for_model(model_like, Xtest_for_model, y_test, label_encoder)
                if roc_data is None:
                    continue
                any_plotted = True
                color = 'darkorange' if series_key == 'original' else 'darkgreen'
                if 'binary' in roc_data:
                    fpr = roc_data['binary']['fpr']; tpr = roc_data['binary']['tpr']; auc_v = roc_data['binary']['auc']
                    _plot_roc_line_with_markers(fpr, tpr, label=f"{series_key} (AUC={auc_v:.3f})", color=color, lw=2, marker_step=10)
                else:
                    if roc_data.get('micro') is not None:
                        fpr = roc_data['micro']['fpr']; tpr = roc_data['micro']['tpr']; auc_v = roc_data['micro']['auc']
                        _plot_roc_line_with_markers(fpr, tpr, label=f"{series_key} micro (AUC={auc_v:.3f})", color=color, lw=2, marker_step=10)
                    else:
                        per = roc_data.get('per_class') or {}
                        for i, clsd in per.items():
                            if clsd is None: continue
                            _plot_roc_line_with_markers(clsd['fpr'], clsd['tpr'], label=f"{series_key} class{i} (AUC={clsd['auc']:.3f})", color=color, lw=1.5, marker_step=10)
                            break
            if not any_plotted:
                plt.close(); continue
            ax.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
            ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'ROC comparison for {model_name} (original vs lemmatized)')
            ax.legend(loc='lower right', fontsize='small')
            plt.tight_layout()
            outp = os.path.join(PLOTS_DIR, f"roc_comparison_{model_name}.png")
            plt.savefig(outp, dpi=150); plt.close()
            print("Saved ROC comparison:", outp)

    total_duration = time.time() - total_start
    # save runtime summary
    runtime_summary = {"total_seconds": float(total_duration), "start_time": None, "end_time": None}
    runtime_summary["total_seconds"] = float(total_duration)
    runtime_path = os.path.join(OUTPUT_DIR, "runtime_summary.json")
    with open(runtime_path, "w", encoding="utf-8") as rf:
        json.dump({"total_seconds": float(total_duration)}, rf, indent=2)
    print(f"Total runtime: {total_duration:.1f}s. Saved runtime summary to {runtime_path}")

if __name__ == "__main__":
    main()
