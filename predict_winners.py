import re, json, sys, warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# file paths for input data
CSV = "tdf_stage_results_2020_2025.csv"
RIDER_DATA = "rider_data.json"
RIDERS_2026_CSV = "2026_riders.csv"
YEAR = 2026 # Year for which predictions are to be made

# Manual elevation data for each stage (in meters)
ELEVATIONS = {
    1: 200, 2: 2500, 3: 3850, 4: 2700, 5: 1600,
    6: 4100, 7: 850, 8: 1150, 9: 3300,
    10: 3800, 11: 1400, 12: 1800, 13: 2400,
    14: 3800, 15: 3950, 16: 500,
    17: 2200, 18: 3900, 19: 3500, 20: 5450, 21: 1000,
}

# type of each stage
STAGE_TYPES = {
    1: 'TTT', 2: 'hilly', 3: 'mountain', 4: 'hilly', 5: 'flat',
    6: 'mountain', 7: 'flat', 8: 'flat', 9: 'hilly',
    10: 'mountain', 11: 'flat', 12: 'flat', 13: 'hilly',
    14: 'mountain', 15: 'mountain', 16: 'ITT',
    17: 'flat', 18: 'mountain', 19: 'mountain', 20: 'mountain', 21: 'flat',
}

# distance of each stage (in kilometers)
STAGE_DISTANCES = {
    1: 19.6, 2: 168.5, 3: 195.9, 4: 181.9, 5: 158.3,
    6: 186.2, 7: 175.1, 8: 180.4, 9: 185.5,
    10: 166.6, 11: 161.3, 12: 179.1, 13: 205.8,
    14: 155.3, 15: 183.9, 16: 26.1,
    17: 174.7, 18: 185.2, 19: 127.9, 20: 170.9, 21: 133.0,
}

# ──────────────────────────────────────────────────────────────────────────
# 1.  LOAD 2026 RIDERS
# ──────────────────────────────────────────────────────────────────────────

# Load the list of riders participating in the 2026 Tour de France
rider_csv = pd.read_csv(RIDERS_2026_CSV)
# Combine first name and last name to create the full rider name
rider_csv['rider_name'] = rider_csv['Name'] + ' ' + rider_csv['LastName']
# Clean extra spaces
rider_csv['rider_name'] = rider_csv['rider_name'].str.replace(r'\s+', ' ', regex=True).str.strip()
# Extract the list of 2026 rider names
riders_2026 = rider_csv['rider_name'].tolist()
print(f"2026 riders: {len(riders_2026)}")

# ──────────────────────────────────────────────────────────────────────────
# 2.  LOAD HISTORICAL DATA & BUILD REFERENCE MAPPINGS
# ──────────────────────────────────────────────────────────────────────────

# Load historical stage results data
hist = pd.read_csv(CSV)
print(f"Historical data: {hist.shape} ({hist['won'].sum()} winners)")

# Helper function to find the most frequent positive value in a series
def best_val(series):
    s = series[series > 0]
    if len(s) > 0:
        return int(s.mode().iloc[0]) if len(s.mode()) > 0 else int(s.iloc[0])
    return 0

# lookup dictionary for rider statistics from historical data
rider_lookup = {}
for _, row in hist.iterrows():
    n = row['rider_name']
    if n not in rider_lookup:
        rider_lookup[n] = {
            'rider_name': n,
            'rider_type': row['rider_type'],
            'pcs_ranking': row['pcs_ranking'],
            'hist_stage_wins': row['hist_stage_wins'],
            'PCS_points': row['PCS_points'],
        }
    else:
        # Update rider stats if better values are found (e.g., lower ranking, more wins)
        r = rider_lookup[n]
        if row['pcs_ranking'] > 0 and (r['pcs_ranking'] == 0 or row['pcs_ranking'] < r['pcs_ranking']):
            r['pcs_ranking'] = row['pcs_ranking']
        if row['hist_stage_wins'] > r['hist_stage_wins']:
            r['hist_stage_wins'] = row['hist_stage_wins']
        if row['PCS_points'] > r['PCS_points']:
            r['PCS_points'] = row['PCS_points']
        if row['rider_type'] != 'Unknown':
            r['rider_type'] = row['rider_type']

# Also load rider_data.json for additional lookups and enrich rider_lookup
with open(RIDER_DATA) as f:
    rider_map = json.load(f)

for key, val in rider_map.items():
    n = val['name']
    if n not in rider_lookup:
        rider_lookup[n] = {
            'rider_name': n,
            'rider_type': 'Unknown', # Default to Unknown if not found in historical data
            'pcs_ranking': val.get('rank', 0),
            'hist_stage_wins': val.get('wins', 0),
            'PCS_points': val.get('pts', 0),
        }

# Normalize function for matching rider names (removes accents, special chars, and converts to lowercase)
def normalize_name(name):
    name = name.lower().strip()
    reps = {'é':'e','è':'e','ê':'e','ë':'e','à':'a','â':'a','î':'i','ï':'i',
            'ô':'o','ö':'o','ù':'u','û':'u','ü':'u','ç':'c','č':'c','ć':'c',
            'š':'s','ş':'s','ž':'z','ź':'z','ł':'l','ń':'n','á':'a','í':'i',
            'ó':'o','ú':'u','ý':'y','ė':'e','ø':'o','æ':'ae','ğ':'g','ő':'o','ű':'u'}
    for a, b in reps.items():
        name = name.replace(a, b)
    return re.sub(r'[^a-z\s-]', '', name).strip()

# normalized name lookup for easier matching
norm_lookup = {}
for n in rider_lookup:
    norm_lookup[normalize_name(n)] = n

# Match 2026 riders with historical data and fill in missing info for unmatched riders
matched_riders = []
unmatched = []
for rn in riders_2026:
    nn = normalize_name(rn)
    if nn in norm_lookup:
        matched_riders.append({**rider_lookup[norm_lookup[nn]], 'rider_name': rider_lookup[norm_lookup[nn]]['rider_name']})
    elif rn in rider_lookup: # Check for exact match if normalized match fails
        matched_riders.append({**rider_lookup[rn], 'rider_name': rn})
    else:
        unmatched.append(rn)
        # If a rider is completely unmatched, initialize with default values
        matched_riders.append({
            'rider_name': rn, 'rider_type': 'Unknown',
            'pcs_ranking': 0, 'hist_stage_wins': 0, 'PCS_points': 0,
        })

rider_df = pd.DataFrame(matched_riders)
if unmatched:
    print(f"Riders that wont be on 2026 edition: {len(unmatched)} ({unmatched[:5]}...)")

# ──────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENCODERS & TRAIN MODEL
# ──────────────────────────────────────────────────────────────────────────

# copy of the historical data for feature engineering
df = hist.copy()
TARGET = 'won'

# Drop irrelevant columns from the training data
for c in ['position', 'gap_pct']:
    if c in df.columns:
        df = df.drop(columns=[c])

# Define categorical columns and apply LabelEncoding
cat_cols = ['rider_name', 'stage_type', 'rider_type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str)) # Handle potential NaN values
    encoders[col] = le # Store encoders for later use with prediction data

# Mapping of rider types to general stage types to determine type_stage_match
type_map = {
    'Sprint': 'flat', 'Classic': 'hilly', 'Hills': 'hilly',
    'Climber': 'mountain', 'GC': 'mountain',
    'Time-trialist': 'ITT', 'Allrounder': 'hilly',
}
# Create a feature indicating if a rider's type matches the stage type
df['type_stage_match'] = df.apply(
    lambda r: int(type_map.get(r['rider_type'], '') == r['stage_type']), axis=1
)

# Apply log transformation to skewed numerical features
log_map = {'pcs_ranking': 'log_pcs_ranking', 'hist_stage_wins': 'log_hist_wins',
           'PCS_points': 'log_pcs_points', 'elevation': 'log_elevation'}
for src, dst in log_map.items():
    df[dst] = np.log1p(df[src].clip(lower=0)) # log1p handles zero values gracefully

# Calculate meters climbed per kilometer (climbing intensity)
df['meters_climbed_per_km'] = df['elevation'] / df['distance_km'].clip(lower=1) # Avoid division by zero
df['log_meters_climbed_per_km'] = np.log1p(df['meters_climbed_per_km'].clip(lower=0))
# Feature to indicate potential for breakaway in flat/hilly stages
df['breakaway_idx'] = df['stage'] * df['stage_type'].isin(['flat', 'hilly']).astype(int)

# Sort data to compute lagged features correctly
df = df.sort_values(['rider_name', 'year'])
# Calculate previous year's wins for each rider
df['prev_year_wins'] = df.groupby('rider_name')['won'].transform(
    lambda s: s.shift(1).rolling(1, min_periods=1).sum().fillna(0)
)
df['log_prev_year_wins'] = np.log1p(df['prev_year_wins'])
# Calculate a 3-year rolling average of wins for each rider
df['wins_3yr_avg'] = df.groupby('rider_name')['won'].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean().fillna(0)
)

# Define the list of features to be used in the model
feature_cols = [
    'rider_name_enc', 'stage_type_enc', 'rider_type_enc',
    'stage', 'year', 'distance_km',
    'log_elevation', 'log_pcs_ranking', 'log_hist_wins', 'log_pcs_points',
    'type_stage_match', 'log_prev_year_wins', 'wins_3yr_avg',
    'log_meters_climbed_per_km', 'breakaway_idx',
]

# Identify numerical columns for scaling
num_cols = [c for c in feature_cols
            if c not in ('rider_name_enc', 'stage_type_enc', 'rider_type_enc', 'type_stage_match')]

# Build name -> encoding mapping from historical data for consistency
name_enc_map = dict(zip(df['rider_name'], df['rider_name_enc']))
stage_type_enc_map = dict(zip(encoders['stage_type'].classes_,
                              encoders['stage_type'].transform(encoders['stage_type'].classes_)))
rider_type_enc_map = dict(zip(encoders['rider_type'].classes_,
                               encoders['rider_type'].transform(encoders['rider_type'].classes_)))

# Sort the DataFrame for XGBRanker group parameter (needs data sorted by group)
df_sorted = df.sort_values(['year', 'stage']).reset_index(drop=True)
df_sorted['group_id'] = df_sorted['year'].astype(str) + '_S' + df_sorted['stage'].astype(str)
df_sorted = df_sorted.sort_values('group_id').reset_index(drop=True)

# Prepare features (X) and target (y) for the model
X = df_sorted[feature_cols].copy()
y = df_sorted[TARGET].values
# Determine group counts for XGBRanker (number of riders per stage in each group)
_, group_counts = np.unique(df_sorted['group_id'].values, return_counts=True)

# Initialize StandardScaler for numerical features
scaler = ColumnTransformer([
    ('num', StandardScaler(), num_cols), # Apply StandardScaler to numerical columns
], remainder='passthrough', verbose_feature_names_out=False) # Keep other columns as is

# Scale the features
X_scaled = scaler.fit_transform(X)

print("Training XGBRanker...")
# Initialize and train the XGBRanker model
model = xgb.XGBRanker(
    objective='rank:ndcg', eval_metric='ndcg@5', # Optimization for ranking performance (NDCG@5)
    random_state=42, verbosity=0, # For reproducibility and quiet training
    n_estimators=200, max_depth=5, learning_rate=0.05, # Hyperparameters
    subsample=0.8, colsample_bytree=0.8, # Regularization to prevent overfitting
)
model.fit(X_scaled, y, group=group_counts, verbose=False) # Train the model with group information

# Get the latest temporal stats (previous year wins, 3yr avg wins) for each rider
latest_stats = df_sorted.sort_values('year').groupby('rider_name').last()
latest_stats = latest_stats[['log_prev_year_wins', 'wins_3yr_avg']].reset_index()

# ──────────────────────────────────────────────────────────────────────────
# 4.  BUILD 2026 PREDICTION DATASET
# ──────────────────────────────────────────────────────────────────────────

# Create a DataFrame for all 2026 stages
stages_2026 = pd.DataFrame([{
    'stage': s, 'year': YEAR,
    'stage_type': STAGE_TYPES[s],
    'distance_km': STAGE_DISTANCES[s],
    'elevation': ELEVATIONS[s],
} for s in range(1, 22)]) # Loop through all 21 stages

# Combine stage and rider data to create prediction dataset (all riders for all stages)
rows = []
for _, stage in stages_2026.iterrows():
    for _, rider in rider_df.iterrows():
        rows.append({**stage.to_dict(), **rider.to_dict()})

pred_df = pd.DataFrame(rows)

# Function for safe encoding of categorical features, handling unseen values
def safe_encode(name, enc_map, default=0):
    if name in enc_map:
        return enc_map[name]
    # Try normalized match if exact match fails
    nn = normalize_name(name)
    for k, v in enc_map.items():
        if normalize_name(k) == nn:
            return v
    return default # Return default if no match is found

# Encode categorical features in the prediction dataset using previously fitted encoders
pred_df['rider_name_enc'] = pred_df['rider_name'].apply(
    lambda n: safe_encode(n, name_enc_map, default=0))
pred_df['stage_type_enc'] = pred_df['stage_type'].apply(
    lambda t: safe_encode(t, stage_type_enc_map, default=0))
pred_df['rider_type_enc'] = pred_df['rider_type'].apply(
    lambda t: safe_encode(t, rider_type_enc_map, default=0))

# Re-create 'type_stage_match' feature for the prediction data
pred_df['type_stage_match'] = pred_df.apply(
    lambda r: int(type_map.get(r['rider_type'], '') == r['stage_type']), axis=1)

# Apply log transformations to numerical features for consistency with training data
for src, dst in log_map.items():
    pred_df[dst] = np.log1p(pred_df[src].clip(lower=0))

# Calculate meters climbed per kilometer and breakaway index for prediction data
pred_df['meters_climbed_per_km'] = pred_df['elevation'] / pred_df['distance_km'].clip(lower=1)
pred_df['log_meters_climbed_per_km'] = np.log1p(pred_df['meters_climbed_per_km'].clip(lower=0))
pred_df['breakaway_idx'] = pred_df['stage'] * pred_df['stage_type'].isin(['flat', 'hilly']).astype(int)

# Merge latest temporal stats (log_prev_year_wins, wins_3yr_avg) into the prediction DataFrame
pred_df['_norm'] = pred_df['rider_name'].apply(normalize_name)
latest_stats['_norm'] = latest_stats['rider_name'].apply(normalize_name)
pred_df = pred_df.merge(latest_stats[['_norm', 'log_prev_year_wins', 'wins_3yr_avg']],
                        on='_norm', how='left', suffixes=('', '_latest'))
# Fill any missing values after merge (e.g., for new riders without historical wins)
pred_df['log_prev_year_wins'] = pred_df['log_prev_year_wins'].fillna(0)
pred_df['wins_3yr_avg'] = pred_df['wins_3yr_avg'].fillna(0)

# ──────────────────────────────────────────────────────────────────────────
# 5.  PREDICT
# ──────────────────────────────────────────────────────────────────────────

# Softmax function to convert raw scores into probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / (e_x.sum() + 1e-15) # small epsilon to avoid division by zero

# Prepare the prediction features and scale them using the trained scaler
pred_X = pred_df[feature_cols].copy()
pred_X_scaled = scaler.transform(pred_X)

# Make predictions using the trained XGBRanker model
pred_scores = model.predict(pred_X_scaled)
pred_df['score'] = pred_scores

# ──────────────────────────────────────────────────────────────────────────
# 6.  OUTPUT RESULTS
# ──────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("TOUR DE FRANCE 2026 — STAGE WINNER PREDICTIONS")
print(f"{'='*80}")

all_results = []
# Iterate through each stage to generate and display top 5 predictions
for s in range(1, 22):
    mask = pred_df['stage'] == s
    sp = pred_df[mask].copy() # Get predictions for the current stage
    sp_scores = sp['score'].values
    sp_probs = softmax(sp_scores) # Convert scores to probabilities
    sp = sp.copy()
    sp['prob'] = sp_probs
    sp = sp.sort_values('prob', ascending=False) # Sort by probability to get top riders

    st = STAGE_TYPES[s]
    d = STAGE_DISTANCES[s]
    e = ELEVATIONS[s]

    print(f"\n{'─'*80}")
    print(f"Stage {s:2d}  |  {st.upper():10s}  |  {d:5.1f}km  |  D+ {e:>4}m")
    print(f"{'─'*80}")

    # Print the top 5 riders for the current stage
    for i in range(5):
        r = sp.iloc[i]
        print(f"  {i+1}. {r['rider_name']:30s}  {r['rider_type']:15s}  "
              f"prob={r['prob']:.4f}  (rank={r['pcs_ranking']:>4d}, "
              f"wins={r['hist_stage_wins']:>3d})")
        # Store results for CSV export
        all_results.append({
            'stage': s, 'stage_type': st,
            'predicted_rank': i + 1,
            'rider_name': r['rider_name'],
            'rider_type': r['rider_type'],
            'probability': round(r['prob'], 4),
            'pcs_ranking': int(r['pcs_ranking']),
            'hist_stage_wins': int(r['hist_stage_wins']),
        })

out_df = pd.DataFrame(all_results)
out_df.to_csv('predictions_2026.csv', index=False) # Save all predictions to a CSV file
print(f"\n{'='*80}")
print("Saved predictions_2026.csv")
print(f"{'='*80}")
