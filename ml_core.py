"""
ml_core.py — CSV ML Compare の計算部分。
ブラウザ内（Pyodide）でも、普通の Python でも同じように動く。

使い方（Python から直接）:
    import ml_core
    info = ml_core.load_csv(open("data.csv").read())
    res  = ml_core.run(target="y", features=[...], task="auto",
                       models=["logreg", "rf", "lgbm"], test_size=0.25, seed=42, cv_folds=0)
"""
import importlib.util
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


# ----------------------------------------------------------------- external boosters (読み込みは使う時だけ)
def _lgbm(task, s):
    import lightgbm as lgb
    kw = dict(n_estimators=300, learning_rate=0.05, random_state=s, n_jobs=1, verbose=-1)
    return lgb.LGBMClassifier(**kw) if task == "classification" else lgb.LGBMRegressor(**kw)


def _xgb(task, s):
    import xgboost as xgb
    kw = dict(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=s, n_jobs=1, tree_method="hist")
    return xgb.XGBClassifier(**kw) if task == "classification" else xgb.XGBRegressor(**kw)


def _cat(task, s):
    import catboost as cb
    kw = dict(iterations=300, learning_rate=0.08, random_seed=s, verbose=0, thread_count=1, allow_writing_files=False)
    return cb.CatBoostClassifier(**kw) if task == "classification" else cb.CatBoostRegressor(**kw)


# ----------------------------------------------------------------- model catalogue
# id: (表示名, 作り方, 必要な追加パッケージ)
MODELS = {
    "classification": {
        "baseline": ("ベースライン（最多クラスを当てる）", lambda s: DummyClassifier(strategy="most_frequent"), None),
        "logreg":   ("ロジスティック回帰", lambda s: LogisticRegression(max_iter=3000), None),
        "tree":     ("決定木", lambda s: DecisionTreeClassifier(random_state=s, min_samples_leaf=5), None),
        "rf":       ("ランダムフォレスト", lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1), None),
        "hgb":      ("勾配ブースティング（sklearn）", lambda s: HistGradientBoostingClassifier(random_state=s), None),
        "lgbm":     ("LightGBM", lambda s: _lgbm("classification", s), "lightgbm"),
        "xgb":      ("XGBoost", lambda s: _xgb("classification", s), "xgboost"),
        "cat":      ("CatBoost", lambda s: _cat("classification", s), "catboost"),
        "knn":      ("k近傍法", lambda s: KNeighborsClassifier(n_neighbors=15), None),
        "svm":      ("サポートベクターマシン", lambda s: SVC(probability=True, random_state=s), None),
        "nb":       ("ナイーブベイズ", lambda s: GaussianNB(), None),
    },
    "regression": {
        "baseline": ("ベースライン（平均値を当てる）", lambda s: DummyRegressor(strategy="mean"), None),
        "linear":   ("線形回帰", lambda s: LinearRegression(), None),
        "ridge":    ("リッジ回帰", lambda s: Ridge(alpha=1.0), None),
        "tree":     ("決定木", lambda s: DecisionTreeRegressor(random_state=s, min_samples_leaf=5), None),
        "rf":       ("ランダムフォレスト", lambda s: RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=1), None),
        "hgb":      ("勾配ブースティング（sklearn）", lambda s: HistGradientBoostingRegressor(random_state=s), None),
        "lgbm":     ("LightGBM", lambda s: _lgbm("regression", s), "lightgbm"),
        "xgb":      ("XGBoost", lambda s: _xgb("regression", s), "xgboost"),
        "cat":      ("CatBoost", lambda s: _cat("regression", s), "catboost"),
        "knn":      ("k近傍法", lambda s: KNeighborsRegressor(n_neighbors=15), None),
        "svr":      ("サポートベクター回帰", lambda s: TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()), None),
    },
}
SLOW = {"svm", "svr"}           # 行数が多いと極端に遅くなるモデル
SLOW_ROWS = 5000
IMP_ROWS = 2000                 # 重要度の計算に使う最大行数（速度のため）
IMP_REPEATS = 5

_DF = None                      # 読み込んだデータ（load_csv で設定）


def _have(pkg):
    return pkg is None or importlib.util.find_spec(pkg) is not None


def catalogue(loadable=()):
    """loadable: まだ入っていないが、必要になったら読み込めるパッケージ名（ブラウザ版で使う）"""
    loadable = set(loadable or ())
    out = {}
    for t, m in MODELS.items():
        out[t] = [{"id": k, "name": v[0], "pkg": v[2],
                   "available": _have(v[2]) or (v[2] in loadable)} for k, v in m.items()]
    return json.dumps(out, ensure_ascii=False)


def required_packages(task, models):
    """選ばれたモデルに必要な追加パッケージ（未インストールのもの）"""
    need = []
    for mid in models or []:
        for t in ([task] if task in MODELS else list(MODELS)):
            pkg = MODELS[t].get(mid, (None, None, None))[2]
            if pkg and not _have(pkg) and pkg not in need:
                need.append(pkg)
    return json.dumps(need)


# ----------------------------------------------------------------- loading
def _read(text):
    text = text.lstrip("﻿")
    best = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
        except Exception:
            continue
        if df.shape[1] > 1 and (best is None or df.shape[1] > best.shape[1]):
            best = df
    if best is None:
        try:
            best = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(f"CSVとして読めませんでした（{e}）")
    if best.shape[1] < 2:
        raise ValueError("列が1つしか見つかりません。区切り文字（カンマ・タブ・セミコロン）を確認してください。")
    return best


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
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "classification"
    if nn.nunique() <= 10 and np.allclose(nn.astype(float), nn.astype(float).round()):
        return "classification"
    return "regression"


def _clean(df):
    df.columns = [str(c).strip() or f"列{i + 1}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        seen, cols = {}, []
        for c in df.columns:
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        df.columns = cols
    df = df.dropna(how="all")
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype(float)
        elif s.dtype == object:
            # "1,234" や " 12 " のような数値の文字列は数値に直す
            t = s.astype(str).str.strip().str.replace(",", "", regex=False)
            num = pd.to_numeric(t.where(s.notna()), errors="coerce")
            if s.notna().sum() > 0 and num.notna().sum() == s.notna().sum():
                df[c] = num
            else:
                low = s.dropna().astype(str).str.strip().str.lower()
                if len(low) and set(low.unique()) <= {"true", "false"}:
                    df[c] = s.map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() == "true"))
                else:
                    df[c] = s.where(s.isna(), s.astype(str).str.strip())
    return df


def load_csv(text):
    global _DF
    df = _clean(_read(text))
    if len(df) == 0:
        raise ValueError("データの行がありません。")
    _DF = df
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": c, "kind": _kind(s), "missing": int(s.isna().sum()),
                     "unique": int(s.nunique()), "task": _suggest_task(s)})
    head = df.head(8)
    preview = head.astype(object).where(head.notna(), "").astype(str).values.tolist()
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
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                       ("oh", ohe)]), cat))
    return ColumnTransformer(parts)


def _fmt(v):
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    return None if not np.isfinite(v) else round(v, 4)


def _short(e):
    s = str(e).strip().split("\n")[-1]
    return s[:200]


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
    for c in features:  # 文字列の列は文字列にそろえる（数字と文字の混在対策）
        if not pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    drop_empty = [c for c in features if data[c].notna().sum() == 0]
    if drop_empty:
        notes.append("中身が空の列を外しました：" + "、".join(drop_empty))
        features = [c for c in features if c not in drop_empty]
        if not features:
            raise ValueError("使える説明の列がありません。")
    if task == "auto":
        task = _suggest_task(data[target])
    X = data[features]
    y_raw = data[target]

    if task == "classification":
        y_str = y_raw.map(lambda v: str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
        classes = sorted(y_str.unique().tolist())
        if len(classes) < 2:
            raise ValueError("予測したい列に種類が1つしかありません。分類できません。")
        if len(classes) > 50:
            notes.append(f"クラスが {len(classes)} 種類あります。回帰の方が合っているかもしれません。")
        code = {c: i for i, c in enumerate(classes)}
        y = y_str.map(code).astype(int)          # 全モデル共通で 0,1,2… に直して学習（XGBoost対策）
        vc = y.value_counts()
        strat = y if vc.min() >= 2 else None
        if strat is None:
            notes.append("1件しかないクラスがあるため、層化せずに分割しました。")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        keep = y.notna()
        if (~keep).any():
            notes.append(f"数値にできない値の {int((~keep).sum())} 行は除きました。")
            X, y = X[keep], y[keep]
        if y.nunique() < 2:
            raise ValueError("予測したい列の値がすべて同じです。")
        classes, strat = None, None
    if len(y) < 20:
        raise ValueError(f"使える行が {len(y)} 行しかありません。最低20行は必要です。")

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        notes.append("クラスの件数が少ないため、層化せずに分割しました。")
    if task == "classification" and ytr.nunique() < 2:
        raise ValueError("学習用の行に1種類のクラスしか入りませんでした。データを増やすか、テストの割合を下げてください。")

    cat = MODELS[task]
    models = [m for m in (models or []) if m in cat]
    if "baseline" not in models:
        models = ["baseline"] + models

    results, preds, probs, importances = [], {}, {}, {}
    imp_idx = Xte.index if len(Xte) <= IMP_ROWS else Xte.sample(IMP_ROWS, random_state=seed).index
    score = "accuracy" if task == "classification" else "r2"
    for mid in models:
        name, make, pkg = cat[mid]
        if not _have(pkg):
            notes.append(f"{name} はこの環境では使えないため、飛ばしました（{pkg} が入っていません）。")
            continue
        if mid in SLOW and len(Xtr) > SLOW_ROWS:
            notes.append(f"{name} は {SLOW_ROWS} 行を超えると時間がかかりすぎるため、飛ばしました。")
            continue
        t0 = time.time()
        try:
            pipe = Pipeline([("prep", _prep(X)), ("model", make(seed))])
            pipe.fit(Xtr, ytr)
            p = np.asarray(pipe.predict(Xte)).ravel()
        except Exception as e:
            notes.append(f"{name} は学習に失敗したので飛ばしました：{_short(e)}")
            continue
        row = {"id": mid, "name": name}
        if task == "classification":
            p = p.astype(int)
            row["accuracy"] = _fmt(accuracy_score(yte, p))
            row["balanced_accuracy"] = _fmt(balanced_accuracy_score(yte, p))
            row["f1_macro"] = _fmt(f1_score(yte, p, average="macro", zero_division=0))
            auc = None
            if mid != "baseline" and hasattr(pipe, "predict_proba"):
                try:
                    pr = np.asarray(pipe.predict_proba(Xte))
                    model_cls = [int(c) for c in pipe.classes_]
                    if len(classes) == 2 and 1 in model_cls:
                        col = pr[:, model_cls.index(1)]
                        probs[mid] = col
                        if yte.nunique() == 2:
                            auc = roc_auc_score((yte == 1).astype(int), col)
                    elif len(classes) > 2 and yte.nunique() > 2:
                        full = np.zeros((len(Xte), len(classes)))
                        for j, c in enumerate(model_cls):
                            full[:, c] = pr[:, j]
                        present = sorted(yte.unique())
                        sub = full[:, present]
                        sub = sub / np.clip(sub.sum(1, keepdims=True), 1e-12, None)
                        auc = roc_auc_score(yte, sub, multi_class="ovr", average="macro", labels=present)
                except Exception:
                    auc = None
            row["roc_auc"] = _fmt(auc)
            row["confusion"] = confusion_matrix(yte, p, labels=list(range(len(classes)))).tolist()
        else:
            row["r2"] = _fmt(r2_score(yte, p))
            row["mae"] = _fmt(mean_absolute_error(yte, p))
            row["rmse"] = _fmt(np.sqrt(mean_squared_error(yte, p)))
        if cv_folds and cv_folds >= 2:
            try:
                if task == "classification" and y.value_counts().min() >= cv_folds:
                    kf = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
                else:
                    kf = KFold(cv_folds, shuffle=True, random_state=seed)
                cv = cross_val_score(Pipeline([("prep", _prep(X)), ("model", make(seed))]), X, y,
                                     cv=kf, scoring=score, error_score="raise")
                row["cv_mean"], row["cv_sd"] = _fmt(cv.mean()), _fmt(cv.std())
            except Exception:
                row["cv_mean"] = row["cv_sd"] = None
        if mid != "baseline":
            try:
                pi = permutation_importance(pipe, Xte.loc[imp_idx], yte.loc[imp_idx], n_repeats=IMP_REPEATS,
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

    if len(results) <= 1:
        raise ValueError("選んだモデルがすべて動きませんでした。" + (" ".join(notes[-3:]) if notes else ""))

    # 予測一覧（テスト用の行）
    table = []
    for i, idx in enumerate(Xte.index):
        r = {"row": int(idx) + 2 if isinstance(idx, (int, np.integer)) else str(idx),
             "actual": classes[int(yte.loc[idx])] if classes else _fmt(yte.loc[idx])}
        for mid in preds:
            v = preds[mid][i]
            r[mid] = classes[int(v)] if classes else _fmt(v)
            if mid in probs:
                r[mid + "__prob"] = _fmt(probs[mid][i])
        table.append(r)

    return json.dumps({
        "task": task, "target": target, "features": features,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "classes": classes, "metric_primary": score,
        "results": results, "importances": importances, "predictions": table,
        "notes": notes, "cv_folds": int(cv_folds or 0),
    }, ensure_ascii=False)
