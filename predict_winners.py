# pip install pandas numpy scikit-learn xgboost matplotlib seaborn
"""
TDF Stage Winner Prediction — XGBRanker with group-based ranking.

Each (year, stage) is a group; riders are ranked within their stage.
Predictions use softmax across riders per stage instead of a global threshold.
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

import xgboost as xgb

# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD
# ═══════════════════════════════════════════════════════════════════════════

CSV = "tdf_stage_results_2020_2025.csv"
df = pd.read_csv(CSV)
TARGET = 'won'

print(f"Shape: {df.shape}")
n_pos = df[TARGET].sum()
n_neg = len(df) - n_pos
print(f"Positive: {n_pos} ({n_pos/len(df)*100:.2f}%)")
print(f"Negative: {n_neg} ({n_neg/len(df)*100:.2f}%)")
print(f"Imbalance ratio: 1 : {n_neg/n_pos:.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

# --- Drop post-race leak ---
leak_cols = ['position', 'gap_pct']
df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore')

# --- Encode categoricals ---
cat_cols = ['rider_name', 'stage_type', 'rider_type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))
    encoders[col] = le

# --- Rider type ↔ stage type match ---
type_map = {
    'Sprint': 'flat', 'Classic': 'hilly', 'Hills': 'hilly',
    'Climber': 'mountain', 'GC': 'mountain',
    'Time-trialist': 'ITT', 'Allrounder': 'hilly',
}
df['type_stage_match'] = df.apply(
    lambda r: int(type_map.get(r['rider_type'], '') == r['stage_type']), axis=1
)

# --- Log transforms ---
log_cols_src = {'pcs_ranking': 'log_pcs_ranking',
                'hist_stage_wins': 'log_hist_wins',
                'PCS_points': 'log_pcs_points',
                'elevation': 'log_elevation'}
for src, dst in log_cols_src.items():
    df[dst] = np.log1p(df[src].clip(lower=0))

# --- Granulated stage profile: meters climbed per km ---
df['meters_climbed_per_km'] = df['elevation'] / df['distance_km'].clip(lower=1)
df['log_meters_climbed_per_km'] = np.log1p(df['meters_climbed_per_km'].clip(lower=0))

# --- Breakaway likelihood: late flat/hilly stages favour breakaways ---
# Stage number × breakaway-friendly terrain (flat/hilly).
# GC contenders don't fight on these days, so the model should lower
# confidence on elite riders and consider outsiders.
df['breakaway_idx'] = df['stage'] * df['stage_type'].isin(['flat', 'hilly']).astype(int)

# --- Temporal features: previous year wins & 3-year rolling average ---
df = df.sort_values(['rider_name', 'year'])
df['prev_year_wins'] = df.groupby('rider_name')['won'].transform(
    lambda s: s.shift(1).rolling(1, min_periods=1).sum().fillna(0)
)
df['log_prev_year_wins'] = np.log1p(df['prev_year_wins'])
df['wins_3yr_avg'] = df.groupby('rider_name')['won'].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean().fillna(0)
)

# --- Final feature set ---
feature_cols = [
    'rider_name_enc', 'stage_type_enc', 'rider_type_enc',
    'stage', 'year',
    'distance_km',
    'log_elevation', 'log_pcs_ranking',
    'log_hist_wins', 'log_pcs_points',
    'type_stage_match',
    'log_prev_year_wins', 'wins_3yr_avg',
    'log_meters_climbed_per_km',
    'breakaway_idx',
]

# Numeric columns to scale (everything except low-cardinality encodings)
num_cols = [c for c in feature_cols
            if c not in ('rider_name_enc', 'stage_type_enc', 'rider_type_enc',
                         'type_stage_match')]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  GROUP DEFINITION  (year + stage = one group/query)
# ═══════════════════════════════════════════════════════════════════════════
#
# XGBRanker ranks riders within each group.  Groups must be contiguous.
# ═══════════════════════════════════════════════════════════════════════════

df['group_id'] = df['year'].astype(str) + '_S' + df['stage'].astype(str)
df = df.sort_values(['group_id']).reset_index(drop=True)

# --- Data quality check: detect stages with no winner ---
winners_per_stage = df.groupby('group_id')[TARGET].sum()
stages_missing_winner = winners_per_stage[winners_per_stage == 0]
if len(stages_missing_winner) > 0:
    print(f"\n⚠ WARNING: {len(stages_missing_winner)} stage(s) have no winner "
          f"(data likely incomplete from scraper):")
    for s in stages_missing_winner.index:
        n_riders = len(df[df['group_id'] == s])
        print(f"  {s}: {n_riders} riders (missing top finishers)")

rider_names = df['rider_name'].values
X = df[feature_cols].copy()
y = df[TARGET].values
group_ids = df['group_id'].values

n_groups = df['group_id'].nunique()
print(f"Groups (stages): {n_groups}")
print(f"Avg riders per stage: {len(df) / n_groups:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  TEMPORAL TRAIN / TEST SPLIT  (no leakage)
# ═══════════════════════════════════════════════════════════════════════════

train_mask = df['year'] <= 2023
test_mask  = df['year'] >= 2024

X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
y_train, y_test = y[train_mask].copy(), y[test_mask].copy()
group_ids_train = group_ids[train_mask]
group_ids_test  = group_ids[test_mask]
rider_names_test = rider_names[test_mask]

# Group sizes for XGBRanker (must match order in data)
_, train_group_counts = np.unique(group_ids_train, return_counts=True)
_, test_group_counts  = np.unique(group_ids_test,  return_counts=True)

print(f"\nTrain: {X_train.shape}  ({y_train.sum()} winners, "
      f"{len(train_group_counts)} stages)")
print(f"Test:  {X_test.shape}   ({y_test.sum()} winners, "
      f"{len(test_group_counts)} stages)")

# ═══════════════════════════════════════════════════════════════════════════
# 5.  RANKING METRICS  (MRR, NDCG@1, NDCG@5)
# ═══════════════════════════════════════════════════════════════════════════

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-15)


def ranking_metrics(y_true, y_score, g_ids):
    """
    Compute MRR, NDCG@1, NDCG@5 per group.
    Each stage has exactly one winner → binary relevance.
    """
    mrr_list, ndcg1_list, ndcg5_list = [], [], []

    for g in np.unique(g_ids):
        mask = (g_ids == g)
        g_true = y_true[mask]
        g_score = y_score[mask]

        g_prob = softmax(g_score)
        order = np.argsort(-g_prob)
        sorted_true = g_true[order]

        winner_pos = np.where(sorted_true == 1)[0]
        if len(winner_pos) == 0:
            continue

        rank = winner_pos[0] + 1

        # MRR
        mrr_list.append(1.0 / rank)

        # NDCG@1
        ndcg1_list.append(1.0 if rank == 1 else 0.0)

        # NDCG@5
        if rank <= 5:
            ndcg5_list.append(1.0 / np.log2(rank + 1))
        else:
            ndcg5_list.append(0.0)

    return {
        'mrr': np.mean(mrr_list) if mrr_list else 0.0,
        'ndcg@1': np.mean(ndcg1_list) if ndcg1_list else 0.0,
        'ndcg@5': np.mean(ndcg5_list) if ndcg5_list else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  GROUP K-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
#
# GroupKFold ensures no stage is split across train and validation folds.
# scale_pos_weight is NOT used — XGBRanker's loss handles the one-winner
# structure natively.
# ═══════════════════════════════════════════════════════════════════════════

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(X_train, y_train, groups=group_ids_train)
):
    print(f"\n{'─'*60}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'─'*60}")

    # Extract fold data
    X_ft = X_train.iloc[train_idx]
    y_ft = y_train[train_idx]
    g_ft = group_ids_train[train_idx]

    X_fv = X_train.iloc[val_idx]
    y_fv = y_train[val_idx]
    g_fv = group_ids_train[val_idx]

    # Ensure groups are contiguous (XGBRanker requirement)
    train_sort = np.argsort(g_ft, kind='stable')
    val_sort   = np.argsort(g_fv, kind='stable')

    X_ft = X_ft.iloc[train_sort]
    y_ft = y_ft[train_sort]
    g_ft = g_ft[train_sort]

    X_fv = X_fv.iloc[val_sort]
    y_fv = y_fv[val_sort]
    g_fv = g_fv[val_sort]

    _, ft_grp_counts = np.unique(g_ft, return_counts=True)
    _, fv_grp_counts = np.unique(g_fv, return_counts=True)

    n_train_winners = y_ft.sum()
    n_val_winners   = y_fv.sum()
    print(f"  Train stages: {len(ft_grp_counts)}  "
          f"(winners: {n_train_winners})")
    print(f"  Val stages:   {len(fv_grp_counts)}  "
          f"(winners: {n_val_winners})")

    # Scale (fit only on training fold)
    scaler = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
    ], remainder='passthrough', verbose_feature_names_out=False)

    X_ft_scaled = scaler.fit_transform(X_ft)
    X_fv_scaled = scaler.transform(X_fv)

    # Train XGBRanker
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        eval_metric='ndcg@5',
        random_state=42,
        verbosity=0,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    model.fit(
        X_ft_scaled, y_ft,
        group=ft_grp_counts,
        eval_set=[(X_fv_scaled, y_fv)],
        eval_group=[fv_grp_counts],
        verbose=False,
    )

    # Predict raw scores → softmax per group → ranking metrics
    y_fv_score = model.predict(X_fv_scaled)
    metrics = ranking_metrics(y_fv, y_fv_score, g_fv)

    fold_metrics.append({
        'fold': fold + 1,
        'mrr': metrics['mrr'],
        'ndcg@1': metrics['ndcg@1'],
        'ndcg@5': metrics['ndcg@5'],
    })

    print(f"  MRR        = {metrics['mrr']:.4f}")
    print(f"  NDCG@1     = {metrics['ndcg@1']:.4f}")
    print(f"  NDCG@5     = {metrics['ndcg@5']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  CV SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("CROSS-VALIDATION SUMMARY — GroupKFold (year × stage)")
print(f"{'═'*60}")
summary = pd.DataFrame(fold_metrics)
print(summary.round(4).to_string(index=False))
print(f"\n  Mean MRR        = {summary['mrr'].mean():.4f} ± {summary['mrr'].std():.4f}")
print(f"  Mean NDCG@1     = {summary['ndcg@1'].mean():.4f} ± {summary['ndcg@1'].std():.4f}")
print(f"  Mean NDCG@5     = {summary['ndcg@5'].mean():.4f} ± {summary['ndcg@5'].std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  HYPERPARAMETER TUNING  (Randomized search with GroupKFold)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("HYPERPARAMETER TUNING (manual randomized search)")
print(f"{'═'*60}")

param_grid = {
    'n_estimators': [50, 80, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0],
    'reg_alpha': [0, 0.1, 1.0],
}

N_TUNE_ITER = 20
rng = np.random.RandomState(42)

best_score = -np.inf
best_params = None
inner_gkf = GroupKFold(n_splits=3)

for i in range(N_TUNE_ITER):
    params = {k: rng.choice(v) for k, v in param_grid.items()}
    inner_scores = []

    for inner_train_idx, inner_val_idx in inner_gkf.split(
        X_train, y_train, groups=group_ids_train
    ):
        X_it = X_train.iloc[inner_train_idx]
        y_it = y_train[inner_train_idx]
        g_it = group_ids_train[inner_train_idx]

        X_iv = X_train.iloc[inner_val_idx]
        y_iv = y_train[inner_val_idx]
        g_iv = group_ids_train[inner_val_idx]

        # Sort by group
        it_sort = np.argsort(g_it, kind='stable')
        iv_sort = np.argsort(g_iv, kind='stable')
        X_it = X_it.iloc[it_sort]; y_it = y_it[it_sort]; g_it = g_it[it_sort]
        X_iv = X_iv.iloc[iv_sort]; y_iv = y_iv[iv_sort]; g_iv = g_iv[iv_sort]

        _, it_grp = np.unique(g_it, return_counts=True)
        _, iv_grp = np.unique(g_iv, return_counts=True)

        scaler = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
        ], remainder='passthrough', verbose_feature_names_out=False)

        X_it_scaled = scaler.fit_transform(X_it)
        X_iv_scaled = scaler.transform(X_iv)

        m = xgb.XGBRanker(
            objective='rank:ndcg',
            eval_metric='ndcg@5',
            random_state=42,
            verbosity=0,
            **params,
        )
        m.fit(X_it_scaled, y_it, group=it_grp, verbose=False)
        score = m.predict(X_iv_scaled)
        met = ranking_metrics(y_iv, score, g_iv)
        inner_scores.append(met['ndcg@5'])

    avg_ndcg5 = np.mean(inner_scores)
    print(f"  Iter {i+1:2d}/{N_TUNE_ITER}  NDCG@5={avg_ndcg5:.4f}  "
          f"lr={params['learning_rate']}  depth={params['max_depth']}  "
          f"n_est={params['n_estimators']}")

    if avg_ndcg5 > best_score:
        best_score = avg_ndcg5
        best_params = params

print(f"\nBest NDCG@5 (inner CV) = {best_score:.4f}")
print("Best params:")
for p, v in best_params.items():
    print(f"  {p} = {v}")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  FINAL TRAINING ON FULL TRAINING SET
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("FINAL TRAINING — full train set (2020–2023)")
print(f"{'═'*60}")

final_scaler = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
], remainder='passthrough', verbose_feature_names_out=False)

X_train_scaled = final_scaler.fit_transform(X_train)
X_test_scaled  = final_scaler.transform(X_test)

final_model = xgb.XGBRanker(
    objective='rank:ndcg',
    eval_metric='ndcg@5',
    random_state=42,
    verbosity=0,
    **best_params,
)

final_model.fit(
    X_train_scaled, y_train,
    group=train_group_counts,
    eval_set=[(X_test_scaled, y_test)],
    eval_group=[test_group_counts],
    verbose=False,
)

# ═══════════════════════════════════════════════════════════════════════════
# 10.  FINAL EVALUATION ON HELD-OUT TEST SET (2024–2025)
# ═══════════════════════════════════════════════════════════════════════════
#
# Instead of a global threshold, we apply softmax per stage and pick the
# rider with the highest relative probability.
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print("FINAL EVALUATION — HELD-OUT TEST SET (2024–2025)")
print(f"{'═'*60}")

y_test_score = final_model.predict(X_test_scaled)
test_metrics = ranking_metrics(y_test, y_test_score, group_ids_test)

print(f"  MRR        = {test_metrics['mrr']:.4f}")
print(f"  NDCG@1     = {test_metrics['ndcg@1']:.4f}")
print(f"  NDCG@5     = {test_metrics['ndcg@5']:.4f}")

# --- Per-stage winner predictions (top-5) ---
print(f"\n{'─'*80}")
print("STAGE WINNER PREDICTIONS — Top-5 per stage")
print(f"{'─'*80}")
correct_k1 = 0
correct_k5 = 0
n_stages = 0
all_predictions = []

for g in np.unique(group_ids_test):
    mask = (group_ids_test == g)
    g_true = y_test[mask]
    g_names = rider_names_test[mask]
    g_score = y_test_score[mask]
    g_prob = softmax(g_score)

    order = np.argsort(-g_prob)
    sorted_true = g_true[order]
    sorted_names = g_names[order]
    sorted_probs = g_prob[order]

    winner_pos = np.where(sorted_true == 1)[0]
    if len(winner_pos) == 0:
        print(f"  {g}: ⚠ NO WINNER IN DATA")
        continue
    winner_rank = winner_pos[0] + 1

    n_stages += 1
    if winner_rank == 1:
        correct_k1 += 1
    if winner_rank <= 5:
        correct_k5 += 1

    print(f"\n{g}  |  Actual: {sorted_names[winner_pos[0]]}  "
          f"({'✓' if winner_rank == 1 else ('#' + str(winner_rank))})")
    for i in range(5):
        mark = ' ← WINNER' if sorted_true[i] == 1 else ''
        print(f"  {i+1}. {sorted_names[i]:30s}  {sorted_probs[i]:.4f}{mark}")

    for i in range(5):
        all_predictions.append({
            'stage': g,
            'rank': i + 1,
            'predicted_rider': sorted_names[i],
            'probability': sorted_probs[i],
            'is_actual_winner': int(sorted_true[i]),
        })

print(f"\n{'─'*50}")
print(f"  Winner in Top 1:  {correct_k1}/{n_stages} = {correct_k1/n_stages*100:.1f}%")
print(f"  Winner in Top 5:  {correct_k5}/{n_stages} = {correct_k5/n_stages*100:.1f}%")

pred_df = pd.DataFrame(all_predictions)
pred_df.to_csv('stage_predictions_2024_2025.csv', index=False)
print(f"\nSaved top-5 predictions to stage_predictions_2024_2025.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 11.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Ranking metrics bar chart ---
metric_names = ['MRR', 'NDCG@1', 'NDCG@5']
metric_values = [test_metrics['mrr'], test_metrics['ndcg@1'], test_metrics['ndcg@5']]
colours = ['#2ecc71', '#3498db', '#9b59b6']
bars = axes[0].bar(metric_names, metric_values, color=colours, width=0.5)
for bar, val in zip(bars, metric_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('Score')
axes[0].set_title('Test Set Ranking Metrics (2024–2025)')
axes[0].grid(True, axis='y', alpha=0.3)

# --- Winner position distribution ---
winner_ranks = []
for g in np.unique(group_ids_test):
    mask = (group_ids_test == g)
    g_true = y_test[mask]
    g_score = y_test_score[mask]
    g_prob = softmax(g_score)
    order = np.argsort(-g_prob)
    sorted_true = g_true[order]
    winner_pos = np.where(sorted_true == 1)[0]
    if len(winner_pos) > 0:
        winner_ranks.append(winner_pos[0] + 1)

max_rank = max(winner_ranks) if winner_ranks else 20
rank_bins = np.arange(1, min(max_rank + 2, 21)) - 0.5
axes[1].hist(winner_ranks, bins=rank_bins, color='#e74c3c', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Winner Rank Position')
axes[1].set_ylabel('Number of Stages')
axes[1].set_title('Where Does the Winner Land in Model\'s Ranking?')
axes[1].set_xticks(range(1, min(max_rank + 1, 21)))
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ranking_metrics.png', dpi=150)
print("\nSaved ranking_metrics.png")

# --- Feature importance ---
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_,
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df.head(12), x='importance', y='feature', palette='viridis')
plt.title('Top 12 Feature Importance — XGBRanker')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("Saved feature_importance.png")

print(f"\n{'═'*60}")
print("DONE")
print(f"{'═'*60}")
