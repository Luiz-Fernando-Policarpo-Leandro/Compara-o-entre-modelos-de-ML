#!/usr/bin/env python3
# main.py — versão FINAL, revisada, blindada e sem erros
# Tudo é salvo dentro da pasta /data

import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*GIL.*")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

sns.set(style="whitegrid")
RND = 42

# --------------------------
# XGBoost (se instalado)
# --------------------------
HAS_XGB = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

# ==========================
# 1. GARANTIR PASTA data/
# ==========================
OUT = "data"
os.makedirs(OUT, exist_ok=True)

def out(name):
    return os.path.join(OUT, name)

# ==========================
# FUNÇÕES AUXILIARES
# ==========================
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(out(name), dpi=200)
    print("[saved]", out(name))
    plt.close(fig)

def ensure_ohe_kwargs():
    try:
        OneHotEncoder(sparse_output=False)
        return {"handle_unknown": "ignore", "sparse_output": False}
    except:
        return {"handle_unknown": "ignore", "sparse": False}


# ==========================
# 2. CARREGAR CSV
# ==========================
DATA = "salary.csv"
if not os.path.exists(DATA):
    print("Erro: coloque salary.csv no mesmo diretório.")
    sys.exit(1)

df = pd.read_csv(DATA)
print("Loaded dataset shape:", df.shape)
print(df.head().to_string(index=False))
print(df.info())

# ==========================
# 3. LIMPEZA ROBUSTA
# ==========================
# Converter tudo para string nas colunas categóricas
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip().replace("?", np.nan)

# Remover fnlwgt (irrelevante)
if "fnlwgt" in df.columns:
    df = df.drop(columns=["fnlwgt"])

# Log-transform
if "capital-gain" in df.columns:
    df["capital-gain-log"] = np.log1p(df["capital-gain"])

if "capital-loss" in df.columns:
    df["capital-loss-log"] = np.log1p(df["capital-loss"])

# Preencher categorias faltantes
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "salary" in cat_cols:
    cat_cols.remove("salary")

for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

# Preencher numéricos
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ==========================
# 4. CRIAR TARGET BINÁRIO
# ==========================
df["salary"] = df["salary"].astype(str).str.strip()
df["salary_bin"] = (df["salary"] == ">50K").astype(int)

print("Target distribution:")
print(df["salary_bin"].value_counts())

# ==========================
# 5. FEATURES
# ==========================
features = [
    "age", "education-num", "hours-per-week",
    "capital-gain-log", "capital-loss-log",
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["salary_bin"]

# ==========================
# 6. TRAIN/TEST SPLIT
# ==========================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RND, stratify=y
)

# ==========================
# 7. PREPROCESSAMENTO
# ==========================
numerical = X.select_dtypes(include=[np.number]).columns.tolist()
categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()

ohe = OneHotEncoder(**ensure_ohe_kwargs())

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", ohe, categorical)
])

preprocess.fit(X_train_raw)
X_train = preprocess.transform(X_train_raw)
X_test = preprocess.transform(X_test_raw)

# Pegar nomes das features
feature_names = []
feature_names += numerical
try:
    feature_names += list(ohe.get_feature_names_out(categorical))
except:
    feature_names += [f"cat_{i}" for i in range(X_train.shape[1] - len(numerical))]

# ==========================
# 8. MODELOS
# ==========================
results = []

# --------------------------
# LINEAR REGRESSION
# --------------------------
lin = LinearRegression()
lin.fit(X_train, y_train)

y_pred_cont = lin.predict(X_test)
y_pred_cls = (y_pred_cont >= 0.5).astype(int)

mae = mean_absolute_error(y_test, y_pred_cont)
mse = mean_squared_error(y_test, y_pred_cont)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_cont)
acc = accuracy_score(y_test, y_pred_cls)

results.append({
    "model": "LinearRegression",
    "accuracy": acc,
    "roc_auc": None,
    "mae": mae,
    "mse": mse,
    "rmse": rmse
})

pd.DataFrame({"feature": feature_names, "coef": lin.coef_}).to_csv(
    out("linear_coefficients.csv"), index=False
)
print("[saved] linear_coefficients.csv")

# gráficos
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(y_test, y_pred_cont, alpha=0.3)
ax.set_xlabel("Real")
ax.set_ylabel("Predito Contínuo")
ax.set_title("Linear Regression — Real x Previsto")
save_fig(fig, "linear_real_vs_pred.png")

residuals = y_test - y_pred_cont
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(residuals, kde=True, ax=ax)
ax.set_title("Resíduos — Linear")
save_fig(fig, "linear_residuals.png")

# --------------------------
# LOGISTIC REGRESSION
# --------------------------
log = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
log.fit(X_train, y_train)

y_pred_log = log.predict(X_test)
y_prob_log = log.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_prob_log)

results.append({
    "model": "LogisticRegression",
    "accuracy": acc,
    "roc_auc": auc_log
})

df_log = pd.DataFrame({
    "feature": feature_names,
    "coef": log.coef_[0],
})
df_log["odds_ratio"] = np.exp(df_log["coef"])
df_log.sort_values(by="odds_ratio", ascending=False).to_csv(
    out("logistic_odds_ratios.csv"), index=False
)
print("[saved] logistic_odds_ratios.csv")

# confusion logistic
cm = confusion_matrix(y_test, y_pred_log)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix — Logistic")
save_fig(fig, "confusion_logistic.png")

# --------------------------
# RANDOM FOREST
# --------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RND,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

results.append({
    "model": "RandomForest",
    "accuracy": acc_rf,
    "roc_auc": auc_rf
})

pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).to_csv(
    out("rf_feature_importances.csv"), index=False
)

# matrix RF
cm = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix — RandomForest")
save_fig(fig, "confusion_rf.png")

# --------------------------
# XGBOOST (se existir)
# --------------------------
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=200,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RND
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:,1]

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    results.append({
        "model": "XGBoost",
        "accuracy": acc_xgb,
        "roc_auc": auc_xgb
    })

    pd.DataFrame({
        "feature": feature_names,
        "importance": xgb.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(
        out("xgb_feature_importances.csv"), index=False
    )

    # confusion
    cm = confusion_matrix(y_test, y_pred_xgb)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix — XGBoost")
    save_fig(fig, "confusion_xgb.png")

# ==========================
# ROC CURVES
# ==========================
roc_entries = []

fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_entries.append(("Logistic", fpr, tpr, roc_auc_score(y_test, y_prob_log)))

fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_entries.append(("RandomForest", fpr, tpr, auc_rf))

if HAS_XGB:
    fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
    roc_entries.append(("XGBoost", fpr, tpr, auc_xgb))

fig, ax = plt.subplots(figsize=(7,6))
for name, fpr, tpr, aucv in roc_entries:
    ax.plot(fpr, tpr, label=f"{name} (AUC={aucv:.4f})")

ax.plot([0,1],[0,1],'k--')
ax.set_title("ROC Curves")
ax.legend()
save_fig(fig, "roc_curves.png")

# ==========================
# SALVAR METRICAS
# ==========================
pd.DataFrame(results).to_csv(out("model_metrics.csv"), index=False)
print("[saved] model_metrics.csv")

# ==========================
# SUMMARY EXECUTIVO
# ==========================
with open(out("summary_executive.txt"), "w", encoding="utf-8") as f:
    f.write("Resumo Executivo — Salary Prediction\n\n")
    f.write(f"Dataset original: '{DATA}'\nShape: {df.shape}\n\n")
    f.write("Distribuição do target:\n")
    f.write(str(df["salary_bin"].value_counts()) + "\n\n")
    f.write("Métricas dos modelos:\n")
    for r in results:
        f.write(str(r) + "\n")

print("[saved] summary_executive.txt")

print("\nTudo pronto. Arquivos gerados dentro de /data/")
