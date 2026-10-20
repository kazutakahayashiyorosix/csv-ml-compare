"""
ml_core.py — CSV ML Compare の計算部分。
ブラウザ内（Pyodide）でも、普通の Python でも同じように動く。

使い方（Python から直接）:
    import ml_core
    info = ml_core.load_csv(open("data.csv").read())
    res  = ml_core.run(target="y", features=[...], task="auto",
                       models=["logreg", "rf"], test_size=0.25, seed=42, cv_folds=0)
"""
import io
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- model catalogue
MODELS = {
    "classification": {
        "baseline": ("ベースライン（最多クラスを当てる）", lambda s: DummyClassifier(strategy="most_frequent")),
        "logreg":   ("ロジスティック回帰", lambda s: LogisticRegression(max_iter=2000)),
        "tree":     ("決定木", lambda s: DecisionTreeClassifier(random_state=s, min_samples_leaf=5)),
        "rf":       ("ランダムフォレスト", lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1)),
        "hgb":      ("勾配ブースティング", lambda s: HistGradientBoostingClassifier(random_state=s)),
        "knn":      ("k近傍法", lambda s: KNeighborsClassifier(n_neighbors=15)),
        "svm":      ("サポートベクターマシン", lambda s: SVC(probability=True, random_state=s)),
        "nb":       ("ナイーブベイズ", lambda s: GaussianNB()),
    },
    "regression": {
        "baseline": ("ベースライン（平均値を当てる）", lambda s: DummyRegressor(strategy="mean")),
        "linear":   ("線形回帰", lambda s: LinearRegression()),
        "ridge":    ("リッジ回帰", lambda s: Ridge(alpha=1.0)),
        "tree":     ("決定木", lambda s: DecisionTreeRegressor(random_state=s, min_samples_leaf=5)),
        "rf":       ("ランダムフォレスト", lambda s: RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=1)),
        "hgb":      ("勾配ブースティング", lambda s: HistGradientBoostingRegressor(random_state=s)),
        "knn":      ("k近傍法", lambda s: KNeighborsRegressor(n_neighbors=15)),
        "svr":      ("サポートベクター回帰", lambda s: TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler())),
    },
}
SLOW = {"svm", "svr"}           # 行数が多いと極端に遅くなるモデル
SLOW_ROWS = 5000
IMP_ROWS = 2000                 # 重要度の計算に使う最大行数（速度のため）

_DF = None                      # 読み込んだデータ（load_csv で設定）


def catalogue():
    return json.dumps({t: [{"id": k, "name": v[0]} for k, v in m.items()] for t, m in MODELS.items()},
                      ensure_ascii=False)


# ----------------------------------------------------------------- loading
def _read(text):
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(io.StringIO(text))


def _kind(s):
    """列の種類: numeric / category / id_like / constant / empty"""
    nn = s.dropna()
    if len(nn) == 0:
        return "empty"
    if nn.nunique() <= 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if nn.nunique() > 0.5 * len(nn) and nn.nunique() > 50:
        return "id_like"
    return "category"


def _suggest_task(s):
    nn = s.dropna()
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    if nn.nunique() <= 10 and np.allclose(nn, nn.round()):
        return "classification"
    return "regression"


def load_csv(text):
    global _DF
    df = _read(text)
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("列名が重複しています。CSVの1行目（列名）を確認してください。")
    _DF = df
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": c,
            "kind": _kind(s),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "task": _suggest_task(s),
        })
    preview = df.head(8).astype(object).where(df.head(8).notna(), "").astype(str).values.tolist()
    return json.dumps({"rows": int(len(df)), "cols": cols, "header": list(df.columns), "preview": preview},
                      ensure_ascii=False)


# ----------------------------------------------------------------- training
def _prep(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False)
    except TypeError:  # 古い scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", ohe)]), cat),
    ])


def _fmt(v):
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, 4)


def run(target, features, task="auto", models=None, test_size=0.25, seed=42, cv_folds=0):
    if _DF is None:
        raise ValueError("先に CSV を読み込んでください。")
    df = _DF
    notes = []
    features = [f for f in features if f != target and f in df.columns]
    if not features:
        raise ValueError("説明に使う列を1つ以上選んでください。")
    if target not in df.columns:
        raise ValueError("予測したい列が見つかりません。")
    data = df[features + [target]].copy()
    n0 = len(data)
    data = data[data[target].notna()]
    if len(data) < n0:
        notes.append(f"予測したい列が空の {n0 - len(data)} 行は除きました。")
    for c in features:  # 文字列の列は文字列にそろえる（混在対策）
        if not pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].astype("string").astype(object).where(data[c].notna(), np.nan)
    if task == "auto":
        task = _suggest_task(data[target])
    X = data[features]
    y = data[target]
    if task == "classification":
        y = y.astype(str)
        vc = y.value_counts()
        if len(vc) < 2:
            raise ValueError("予測したい列に種類が1つしかありません。分類できません。")
        if len(vc) > 50:
            notes.append(f"クラスが {len(vc)} 種類あります。回帰の方が合っているかもしれません。")
        strat = y if vc.min() >= 2 else None
        if strat is None:
            notes.append("1件しかないクラスがあるため、層化せずに分割しました。")
    else:
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors="coerce")
            keep = y.notna()
            if (~keep).any():
                notes.append(f"数値にできない値の {int((~keep).sum())} 行は除きました。")
            X, y = X[keep], y[keep]
        strat = None
    if len(y) < 20:
        raise ValueError(f"使える行が {len(y)} 行しかありません。最低20行は必要です。")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    cat = MODELS[task]
    models = [m for m in (models or []) if m in cat]
    if "baseline" not in models:
        models = ["baseline"] + models
    classes = sorted(y.unique().tolist()) if task == "classification" else None

    results, preds, probs, importances = [], {}, {}, {}
    imp_idx = Xte.index if len(Xte) <= IMP_ROWS else Xte.sample(IMP_ROWS, random_state=seed).index
    for mid in models:
        name, make = cat[mid]
        if mid in SLOW and len(Xtr) > SLOW_ROWS:
            notes.append(f"{name} は {SLOW_ROWS} 行を超えると時間がかかりすぎるため、飛ばしました。")
            continue
        t0 = time.time()
        pipe = Pipeline([("prep", _prep(X)), ("model", make(seed))])
        try:
            pipe.fit(Xtr, ytr)
        except Exception as e:
            notes.append(f"{name} は学習に失敗しました：{e}")
            continue
        p = pipe.predict(Xte)
        row = {"id": mid, "name": name}
        if task == "classification":
            row["accuracy"] = _fmt(accuracy_score(yte, p))
            row["balanced_accuracy"] = _fmt(balanced_accuracy_score(yte, p))
            row["f1_macro"] = _fmt(f1_score(yte, p, average="macro"))
            auc = None
            if hasattr(pipe, "predict_proba") and mid != "baseline":
                try:
                    pr = pipe.predict_proba(Xte)
                    if len(classes) == 2:
                        pos = list(pipe.classes_).index(classes[1])
                        auc = roc_auc_score((yte == classes[1]).astype(int), pr[:, pos])
                        probs[mid] = pr[:, pos]
                    else:
                        auc = roc_auc_score(yte, pr, multi_class="ovr", average="macro", labels=list(pipe.classes_))
                except Exception:
                    auc = None
            row["roc_auc"] = _fmt(auc)
            row["confusion"] = confusion_matrix(yte, p, labels=classes).tolist()
            score = "accuracy"
        else:
            row["r2"] = _fmt(r2_score(yte, p))
            row["mae"] = _fmt(mean_absolute_error(yte, p))
            row["rmse"] = _fmt(np.sqrt(mean_squared_error(yte, p)))
            score = "r2"
        if cv_folds and cv_folds >= 2:
            if task == "classification" and strat is not None and y.value_counts().min() >= cv_folds:
                kf = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
            else:
                kf = KFold(cv_folds, shuffle=True, random_state=seed)
            try:
                cv = cross_val_score(Pipeline([("prep", _prep(X)), ("model", make(seed))]), X, y, cv=kf, scoring=score)
                row["cv_mean"], row["cv_sd"] = _fmt(cv.mean()), _fmt(cv.std())
            except Exception:
                row["cv_mean"] = row["cv_sd"] = None
        if mid != "baseline":
            try:
                pi = permutation_importance(pipe, Xte.loc[imp_idx], yte.loc[imp_idx], n_repeats=5,
                                            random_state=seed, scoring=score)
                importances[mid] = sorted(
                    [{"feature": f, "mean": _fmt(m), "sd": _fmt(s)}
                     for f, m, s in zip(features, pi.importances_mean, pi.importances_std)],
                    key=lambda d: -(d["mean"] or 0))
            except Exception:
                pass
        row["seconds"] = round(time.time() - t0, 2)
        results.append(row)
        preds[mid] = p

    # 予測一覧（テスト用の行）
    table = []
    for i, idx in enumerate(Xte.index):
        r = {"row": int(idx) + 2, "actual": str(yte.loc[idx])}   # +2 = CSVの行番号（1行目は列名）
        for mid in preds:
            v = preds[mid][i]
            r[mid] = str(v) if task == "classification" else _fmt(v)
            if mid in probs:
                r[mid + "__prob"] = _fmt(probs[mid][i])
        table.append(r)

    return json.dumps({
        "task": task, "target": target, "features": features,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "classes": classes, "metric_primary": "accuracy" if task == "classification" else "r2",
        "results": results, "importances": importances, "predictions": table,
        "notes": notes, "cv_folds": int(cv_folds or 0),
    }, ensure_ascii=False)
