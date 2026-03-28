"""
Step 4: Train goal margin predictor and AH cover classifier.
Uses time-based validation (train on 2019-2023, test on 2023-2025).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def get_feature_columns():
    """Feature columns for model training."""
    return [
        # Home team attacking
        'home_goals_scored_h6',
        'home_shots_on_target_h6',
        'home_shots_h6',
        'home_xg_h6',
        
        # Home team defensive
        'home_goals_conceded_h6',
        
        # Away team attacking
        'away_goals_scored_a6',
        'away_shots_on_target_a6',
        'away_shots_a6',
        'away_xg_a6',
        
        # Away team defensive
        'away_goals_conceded_a6',
        
        # Form
        'home_form_pts',
        'away_form_pts',
        
        # Head to head
        'h2h_avg_home_goals',
        'h2h_avg_away_goals',
        'h2h_home_wins',
        
        # Market signals
        'market_home_implied',
        'market_away_implied',
        'market_draw_implied',
        
        # Season context
        'match_week',
        'is_early_season',
        'is_late_season',
    ]


def get_ah_feature_columns():
    """Feature columns for AH classifier (includes AH line)."""
    return get_feature_columns() + ['market_ah_line']


def train_margin_model(train_df, feature_cols):
    """Train goal margin regression model."""
    
    # Prepare data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['goal_margin']
    
    # Fill NaN with median for features
    for col in feature_cols:
        if X_train[col].isna().any():
            X_train[col] = X_train[col].fillna(X_train[col].median())
    
    # Train model
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_ah_classifier(train_df, feature_cols):
    """Train AH cover probability classifier."""
    
    # Filter to valid AH outcomes (not push)
    ah_df = train_df[train_df['ah_home_covers'].isin([0, 1])].copy()
    
    # Prepare data
    X_train = ah_df[feature_cols].copy()
    y_train = ah_df['ah_home_covers'].astype(int)
    
    # Fill NaN with median
    for col in feature_cols:
        if X_train[col].isna().any():
            X_train[col] = X_train[col].fillna(X_train[col].median())
    
    # Train model
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_models(model_margin, model_ah, test_df, feature_cols, ah_feature_cols):
    """Evaluate models on test set."""
    
    # Prepare test data
    X_test = test_df[feature_cols].copy()
    y_test_margin = test_df['goal_margin']
    
    for col in feature_cols:
        if X_test[col].isna().any():
            X_test[col] = X_test[col].fillna(X_test[col].median())
    
    # Margin predictions
    y_pred_margin = model_margin.predict(X_test)
    
    mae = mean_absolute_error(y_test_margin, y_pred_margin)
    rmse = np.sqrt(mean_squared_error(y_test_margin, y_pred_margin))
    
    print("\n=== MARGIN MODEL EVALUATION ===")
    print(f"MAE: {mae:.3f} goals")
    print(f"RMSE: {rmse:.3f} goals")
    print(f"Baseline (always predict 0): MAE = {mean_absolute_error(y_test_margin, np.zeros_like(y_test_margin)):.3f}")
    
    # Compare directional accuracy
    pred_sign = np.sign(y_pred_margin)
    actual_sign = np.sign(y_test_margin)
    # Exclude draws for directional accuracy
    non_draw_mask = actual_sign != 0
    directional_acc = (pred_sign[non_draw_mask] == actual_sign[non_draw_mask]).mean()
    print(f"Directional accuracy (excl draws): {directional_acc:.1%}")
    
    # AH classifier
    ah_test = test_df[test_df['ah_home_covers'].isin([0, 1])].copy()
    X_test_ah = ah_test[ah_feature_cols].copy()
    y_test_ah = ah_test['ah_home_covers'].astype(int)
    
    for col in ah_feature_cols:
        if X_test_ah[col].isna().any():
            X_test_ah[col] = X_test_ah[col].fillna(X_test_ah[col].median())
    
    y_pred_ah_proba = model_ah.predict_proba(X_test_ah)[:, 1]
    y_pred_ah = (y_pred_ah_proba > 0.5).astype(int)
    
    ah_accuracy = accuracy_score(y_test_ah, y_pred_ah)
    ah_auc = roc_auc_score(y_test_ah, y_pred_ah_proba)
    
    print("\n=== AH COVER CLASSIFIER EVALUATION ===")
    print(f"Accuracy: {ah_accuracy:.1%}")
    print(f"AUC: {ah_auc:.3f}")
    print(f"Baseline (always predict 1): {y_test_ah.mean():.1%}")
    
    return {
        'margin_mae': mae,
        'margin_rmse': rmse,
        'directional_acc': directional_acc,
        'ah_accuracy': ah_accuracy,
        'ah_auc': ah_auc,
        'y_pred_margin': y_pred_margin,
        'y_pred_ah_proba': y_pred_ah_proba
    }


def get_feature_importance(model, feature_cols, top_n=15):
    """Get feature importance from model."""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance.head(top_n)


if __name__ == "__main__":
    # Load features
    data_path = Path(__file__).parent.parent / 'data' / 'features_engineered.parquet'
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df)} matches with features")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Time-based split: train on 2019-2023, test on 2024-2025
    train_cutoff = pd.Timestamp('2024-01-01')
    train_df = df[df['Date'] < train_cutoff]
    test_df = df[df['Date'] >= train_cutoff]
    
    print(f"\nTrain set: {len(train_df)} matches (before {train_cutoff.date()})")
    print(f"Test set: {len(test_df)} matches (from {train_cutoff.date()})")
    
    # Get feature columns
    feature_cols = get_feature_columns()
    ah_feature_cols = get_ah_feature_columns()
    
    print(f"\nFeatures for margin model: {len(feature_cols)}")
    print(f"Features for AH model: {len(ah_feature_cols)}")
    
    # Train models
    print("\nTraining margin predictor...")
    model_margin = train_margin_model(train_df, feature_cols)
    
    print("Training AH cover classifier...")
    model_ah = train_ah_classifier(train_df, ah_feature_cols)
    
    # Evaluate
    results = evaluate_models(model_margin, model_ah, test_df, feature_cols, ah_feature_cols)
    
    # Feature importance
    print("\n=== MARGIN MODEL FEATURE IMPORTANCE ===")
    importance_margin = get_feature_importance(model_margin, feature_cols)
    print(importance_margin.to_string(index=False))
    
    print("\n=== AH MODEL FEATURE IMPORTANCE ===")
    importance_ah = get_feature_importance(model_ah, ah_feature_cols)
    print(importance_ah.to_string(index=False))
    
    # Save models
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / 'margin_predictor.pkl', 'wb') as f:
        pickle.dump({'model': model_margin, 'features': feature_cols}, f)
    
    with open(models_dir / 'ah_cover_classifier.pkl', 'wb') as f:
        pickle.dump({'model': model_ah, 'features': ah_feature_cols}, f)
    
    print(f"\nModels saved to {models_dir}")
    
    # Save feature importance
    importance_margin.to_csv(models_dir / 'importance_margin.csv', index=False)
    importance_ah.to_csv(models_dir / 'importance_ah.csv', index=False)
