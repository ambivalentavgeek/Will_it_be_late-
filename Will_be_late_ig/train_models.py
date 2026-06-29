"""
Will It Be Late? — Model Training Script
Trains and compares: Logistic Regression, Linear Regression (as classifier),
Random Forest, XGBoost on the Zomato delivery dataset.
Saves best model + label encoders + scaler to disk.
"""

import pandas as pd
import numpy as np
import pickle
import json
from math import radians, cos, sin, asin, sqrt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1. LOAD & CLEAN
# ──────────────────────────────────────────────

df = pd.read_csv('Zomato_Dataset.csv')

# Strip whitespace from string cols
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# Replace literal 'nan' strings with NaN
df.replace('nan', np.nan, inplace=True)
df.replace('NaN', np.nan, inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Delivery distance (km)
df['delivery_distance_km'] = df.apply(
    lambda r: haversine(r['Restaurant_latitude'], r['Restaurant_longitude'],
                        r['Delivery_location_latitude'], r['Delivery_location_longitude'])
    if not any(pd.isna([r['Restaurant_latitude'], r['Restaurant_longitude'],
                        r['Delivery_location_latitude'], r['Delivery_location_longitude']]))
    else np.nan,
    axis=1
)

# Pickup wait time (minutes)
def time_diff(t1, t2):
    try:
        h1, m1 = map(int, str(t1).split(':'))
        h2, m2 = map(int, str(t2).split(':'))
        diff = (h2 * 60 + m2) - (h1 * 60 + m1)
        return diff if diff >= 0 else diff + 1440
    except:
        return np.nan

df['pickup_wait_min'] = df.apply(
    lambda r: time_diff(r['Time_Orderd'], r['Time_Order_picked']), axis=1
)

# Order hour (time of day)
def extract_hour(t):
    try:
        return int(str(t).split(':')[0])
    except:
        return np.nan

df['order_hour'] = df['Time_Orderd'].apply(extract_hour)
df['is_peak_hour'] = df['order_hour'].apply(
    lambda h: 1 if h is not None and (12 <= h <= 14 or 19 <= h <= 22) else 0
)

# Date features
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
df['order_day_of_week'] = df['Order_Date'].dt.dayofweek  # 0=Mon
df['is_weekend'] = df['order_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Food preparation complexity (domain knowledge)
food_complexity = {'Meal': 3, 'Buffet': 4, 'Snack': 2, 'Drinks': 1}
df['food_complexity'] = df['Type_of_order'].map(food_complexity).fillna(2)

print(f"\nEngineered features - Distance stats:\n{df['delivery_distance_km'].describe()}")


# ──────────────────────────────────────────────
# 3. TARGET VARIABLE — "Late" = above 75th pct
# ──────────────────────────────────────────────

threshold = int(df['Time_taken (min)'].quantile(0.75))
df['is_late'] = (df['Time_taken (min)'] > threshold).astype(int)
print(f"\nLate threshold: {threshold} min")
print(f"Late orders: {df['is_late'].sum()} / {len(df)} ({df['is_late'].mean()*100:.1f}%)")


# ──────────────────────────────────────────────
# 4. ENCODE CATEGORICALS
# ──────────────────────────────────────────────

cat_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
            'Type_of_vehicle', 'Festival', 'City']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown'))
    encoders[col] = le

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Save threshold
with open('threshold.json', 'w') as f:
    json.dump({'late_threshold_min': threshold}, f)


# ──────────────────────────────────────────────
# 5. FEATURE MATRIX
# ──────────────────────────────────────────────

feature_cols = [
    'Delivery_person_Age', 'Delivery_person_Ratings',
    'delivery_distance_km', 'pickup_wait_min',
    'Vehicle_condition', 'multiple_deliveries',
    'food_complexity', 'order_hour', 'is_peak_hour',
    'order_day_of_week', 'is_weekend',
    'Weather_conditions_enc', 'Road_traffic_density_enc',
    'Type_of_order_enc', 'Type_of_vehicle_enc',
    'Festival_enc', 'City_enc'
]

X = df[feature_cols].copy()
y = df['is_late']

# Fill remaining NaNs with median
X.fillna(X.median(), inplace=True)

print(f"\nFeature matrix shape: {X.shape}")

# Save feature names
with open('feature_names.json', 'w') as f:
    json.dump(feature_cols, f)


# ──────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# ──────────────────────────────────────────────
# 7. TRAIN & COMPARE MODELS
# ──────────────────────────────────────────────

results = {}

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_prob = lr.predict_proba(X_test_sc)[:, 1]
results['Logistic Regression'] = {
    'model': lr, 'pred': lr_pred, 'prob': lr_prob, 'scaled': True,
    'accuracy': accuracy_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'roc_auc': roc_auc_score(y_test, lr_prob),
    'cm': confusion_matrix(y_test, lr_pred).tolist()
}

# --- Linear Regression (as classifier, threshold=0.5) ---
linreg = LinearRegression()
linreg.fit(X_train_sc, y_train)
linreg_prob = linreg.predict(X_test_sc)
linreg_pred = (linreg_prob >= 0.5).astype(int)
results['Linear Regression'] = {
    'model': linreg, 'pred': linreg_pred, 'prob': np.clip(linreg_prob, 0, 1), 'scaled': True,
    'accuracy': accuracy_score(y_test, linreg_pred),
    'f1': f1_score(y_test, linreg_pred),
    'precision': precision_score(y_test, linreg_pred),
    'recall': recall_score(y_test, linreg_pred),
    'roc_auc': roc_auc_score(y_test, np.clip(linreg_prob, 0, 1)),
    'cm': confusion_matrix(y_test, linreg_pred).tolist()
}

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'model': rf, 'pred': rf_pred, 'prob': rf_prob, 'scaled': False,
    'accuracy': accuracy_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_prob),
    'cm': confusion_matrix(y_test, rf_pred).tolist()
}

# --- XGBoost ---
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=42, eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'model': xgb, 'pred': xgb_pred, 'prob': xgb_prob, 'scaled': False,
    'accuracy': accuracy_score(y_test, xgb_pred),
    'f1': f1_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred),
    'recall': recall_score(y_test, xgb_pred),
    'roc_auc': roc_auc_score(y_test, xgb_prob),
    'cm': confusion_matrix(y_test, xgb_pred).tolist()
}


# ──────────────────────────────────────────────
# 8. PRINT COMPARISON TABLE
# ──────────────────────────────────────────────

print("\n" + "="*70)
print(f"{'Model':<22} {'Accuracy':>9} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>7}")
print("="*70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['roc_auc']:>7.4f}")
print("="*70)


# ──────────────────────────────────────────────
# 9. FEATURE IMPORTANCES (XGBoost)
# ──────────────────────────────────────────────

xgb_importance = xgb.feature_importances_
fi_dict = dict(zip(feature_cols, [round(float(v), 4) for v in xgb_importance]))
fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
print("\nXGBoost Feature Importances:")
for feat, imp in fi_sorted.items():
    print(f"  {feat:<35} {imp:.4f}")

with open('feature_importance.json', 'w') as f:
    json.dump(fi_sorted, f)


# ──────────────────────────────────────────────
# 10. SAVE METRICS & BEST MODEL
# ──────────────────────────────────────────────

metrics_out = {}
for name, r in results.items():
    metrics_out[name] = {
        'accuracy': round(r['accuracy'], 4),
        'f1': round(r['f1'], 4),
        'precision': round(r['precision'], 4),
        'recall': round(r['recall'], 4),
        'roc_auc': round(r['roc_auc'], 4),
        'confusion_matrix': r['cm']
    }

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_out, f)

# Best model by F1 (XGBoost expected)
best_name = max(results, key=lambda n: results[n]['f1'])
best = results[best_name]
print(f"\n✅ Best model: {best_name} (F1={best['f1']:.4f})")

with open('best_model.pkl', 'wb') as f:
    pickle.dump({'model': best['model'], 'name': best_name, 'scaled': best['scaled']}, f)

print("\n✅ Saved: best_model.pkl, encoders.pkl, scaler.pkl, feature_names.json")
print("✅ Saved: model_metrics.json, feature_importance.json, threshold.json")
print("\nTraining complete!")
