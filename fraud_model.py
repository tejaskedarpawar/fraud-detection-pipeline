import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

# Load data saved in Phase 1
X_train, X_test, y_train, y_test = joblib.load('phase1_data.pkl')

print("Train shape:", X_train.shape)
print("Fraud in train:", y_train.sum(), "out of", len(y_train))

print("Before SMOTE:", y_train.value_counts().to_dict())

smote = SMOTE(random_state=42, sampling_strategy=0.1)
# sampling_strategy=0.1 means fraud will be 10% of legit (not 50%)
# This is more realistic than full 50/50 balancing

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("After SMOTE: ", dict(zip(*np.unique(y_train_sm, return_counts=True))))
print(f"\nNew fraud samples added: {y_train_sm.sum() - y_train.sum()}")


# scale_pos_weight handles residual imbalance after SMOTE
neg = (y_train_sm == 0).sum()
pos = (y_train_sm == 1).sum()
scale = neg / pos

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,
    use_label_encoder=False,
    eval_metric='aucpr',      # area under precision-recall curve
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("XGBoost training complete!")

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],
)

print("LightGBM training complete!")

def evaluate_model(model, X_test, y_test, name):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    roc_auc     = roc_auc_score(y_test, y_prob)
    pr_auc      = average_precision_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC Score:              {roc_auc:.4f}")
    print(f"  Precision-Recall AUC:       {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit','Fraud'],
                yticklabels=['Legit','Fraud'])
    plt.title(f'{name} — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    return y_prob

xgb_probs = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
lgb_probs = evaluate_model(lgb_model, X_test, y_test, "LightGBM")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
RocCurveDisplay.from_predictions(y_test, xgb_probs, ax=axes[0],
    name='XGBoost', color='steelblue')
RocCurveDisplay.from_predictions(y_test, lgb_probs, ax=axes[0],
    name='LightGBM', color='crimson')
axes[0].set_title('ROC Curve')
axes[0].plot([0,1],[0,1], 'k--', alpha=0.4)

# Precision-Recall Curve (more informative for imbalanced data)
PrecisionRecallDisplay.from_predictions(y_test, xgb_probs, ax=axes[1],
    name='XGBoost', color='steelblue')
PrecisionRecallDisplay.from_predictions(y_test, lgb_probs, ax=axes[1],
    name='LightGBM', color='crimson')
axes[1].set_title('Precision-Recall Curve')

plt.tight_layout()
plt.show()

import pandas as pd

feat_imp = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(15).plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('XGBoost — Top 15 Most Important Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 features:\n", feat_imp.head(5))


iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.002,   # ~0.17% fraud rate we observed in EDA
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train)   # train on original data, no SMOTE needed

# IsolationForest returns -1 for anomaly, 1 for normal — convert to 0/1
iso_pred = iso_forest.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

print("Isolation Forest Results:")
print(classification_report(y_test, iso_pred, target_names=['Legit','Fraud']))

# Compare PR-AUC scores and save the winner
xgb_prauc = average_precision_score(y_test, xgb_probs)
lgb_prauc  = average_precision_score(y_test, lgb_probs)

best_model = xgb_model if xgb_prauc >= lgb_prauc else lgb_model
best_name  = "XGBoost" if xgb_prauc >= lgb_prauc else "LightGBM"

joblib.dump(best_model, 'fraud_model.pkl')
print(f"Best model: {best_name} (PR-AUC: {max(xgb_prauc, lgb_prauc):.4f})")
print("Saved to fraud_model.pkl — ready for Phase 3!")