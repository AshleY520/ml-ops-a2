"""
Feature Engineering Pipeline
Creates two different versions of features from the clean base data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_clean_data():
    """Load the cleaned base data"""
    try:
        df = pd.read_parquet("artifacts/athletes_clean.parquet")
        logger.info(f"Loaded clean data: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error("Clean data not found. Please run ingest.py first.")
        raise

def create_feature_version_1(df):
    """
    Feature Version 1: Basic Features
    Minimal feature engineering, mostly raw cleaned features
    """
    logger.info("Creating Feature Version 1: Basic Features")
    
    df_v1 = df.copy()
    
    # Select basic features only
    basic_features = [
        'athlete_id', 'age', 'weight', 'height', 'gender',
        'deadlift', 'candj', 'snatch', 'backsq', 'total_lift',
        'experience', 'schedule',
        'event_timestamp', 'created_timestamp'
    ]
    
    # Keep only available columns
    available_features = [col for col in basic_features if col in df_v1.columns]
    df_v1 = df_v1[available_features]
    
    # Minimal encoding for categorical variables
    df_v1['gender_encoded'] = df_v1['gender'].map({'Male': 1, 'Female': 0})
    
    logger.info(f"V1 features created: {len(df_v1.columns)} columns")
    return df_v1

def clean_engineered_features(df):
    """Clean engineered features to remove inf and extreme values"""
    logger.info("Cleaning engineered features...")
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f"Found infinity values in columns: {inf_cols}")
    
    ratio_cols = [col for col in df.columns if any(x in col.lower() for x in ['ratio', 'per_kg', 'bmi'])]
    for col in ratio_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            upper_bound = df[col].quantile(0.99)
            lower_bound = df[col].quantile(0.01)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def create_feature_version_2(df):
    """
    Feature Version 2: Engineered Features
    Advanced feature engineering with derived metrics
    """
    logger.info("Creating Feature Version 2: Engineered Features")
    
    df_v2 = df.copy()
    
    # Start with all basic features
    df_v2['gender_encoded'] = df_v2['gender'].map({'Male': 1, 'Female': 0})
    
    # Body composition features 
    df_v2['bmi'] = np.where(df_v2['height'] > 0, 
                           df_v2['weight'] / (df_v2['height'] / 100) ** 2, 
                           np.nan)
    df_v2['height_weight_ratio'] = np.where(df_v2['weight'] > 0,
                                           df_v2['height'] / df_v2['weight'],
                                           np.nan)
    
    # Strength performance ratios 
    df_v2['strength_to_weight_ratio'] = np.where(df_v2['weight'] > 0,
                                                 df_v2['total_lift'] / df_v2['weight'],
                                                 np.nan)
    df_v2['deadlift_to_weight_ratio'] = np.where(df_v2['weight'] > 0,
                                                 df_v2['deadlift'] / df_v2['weight'],
                                                 np.nan)
    df_v2['squat_to_deadlift_ratio'] = np.where(df_v2['deadlift'] > 0,
                                               df_v2['backsq'] / df_v2['deadlift'],
                                               np.nan)
    
    # Olympic vs Powerlifting metrics
    df_v2['olympic_lifts_total'] = df_v2['candj'] + df_v2['snatch']
    df_v2['powerlifting_total'] = df_v2['deadlift'] + df_v2['backsq']
    df_v2['olympic_to_powerlifting_ratio'] = np.where(df_v2['powerlifting_total'] > 0,
                                                     df_v2['olympic_lifts_total'] / df_v2['powerlifting_total'],
                                                     np.nan)
    
    # Performance per unit body weight 
    df_v2['deadlift_per_kg'] = np.where(df_v2['weight'] > 0,
                                       df_v2['deadlift'] / df_v2['weight'],
                                       np.nan)
    df_v2['snatch_per_kg'] = np.where(df_v2['weight'] > 0,
                                     df_v2['snatch'] / df_v2['weight'],
                                     np.nan)
    df_v2['candj_per_kg'] = np.where(df_v2['weight'] > 0,
                                    df_v2['candj'] / df_v2['weight'],
                                    np.nan)
    
    # Experience and training features 
    unique_exp = df_v2['experience'].unique()
    logger.info(f"Unique experience values: {unique_exp}")
    
    def map_experience(val):
        if pd.isna(val):
            return 0
        val_str = str(val).strip()
        if val_str in ['1', '1.0']:
            return 1
        elif val_str in ['2', '2.0']:
            return 2
        elif val_str in ['3', '3.0']:
            return 3
        elif val_str in ['4', '4.0']:
            return 4
        elif val_str in ['5+', '5', '5.0']:
            return 5
        else:
            return 0
    
    df_v2['experience_numeric'] = df_v2['experience'].apply(map_experience)
    
    unique_schedule = df_v2['schedule'].unique()
    logger.info(f"Unique schedule values: {unique_schedule}")
    
    def map_schedule(val):
        if pd.isna(val):
            return 0
        val_str = str(val).strip()
        if val_str in ['1', '1.0']:
            return 1
        elif val_str in ['2', '2.0']:
            return 2
        elif val_str in ['3', '3.0']:
            return 3
        elif val_str in ['4', '4.0']:
            return 4
        elif val_str in ['5', '5.0']:
            return 5
        elif val_str in ['6+', '6', '6.0']:
            return 6
        else:
            return 0
    
    df_v2['training_frequency'] = df_v2['schedule'].apply(map_schedule)
    
    # Age categories 
    df_v2['age_category'] = pd.cut(df_v2['age'], 
                                  bins=[0, 25, 30, 35, 100], 
                                  labels=['young', 'mid', 'experienced', 'veteran'],
                                  include_lowest=True)
    df_v2['age_category_encoded'] = df_v2['age_category'].cat.codes
    
    # Weight categories 
    df_v2['weight_category'] = pd.cut(df_v2['weight'],
                                     bins=[0, 60, 70, 80, 90, 200],
                                     labels=['light', 'medium', 'heavy', 'very_heavy', 'super_heavy'],
                                     include_lowest=True)
    df_v2['weight_category_encoded'] = df_v2['weight_category'].cat.codes
    
    df_v2 = clean_engineered_features(df_v2)
    
    logger.info(f"V2 features created: {len(df_v2.columns)} columns")
    logger.info(f"New engineered features: {len(df_v2.columns) - len(df.columns)}")
    
    return df_v2

def save_feature_versions(df_v1, df_v2):
    """Save both feature versions"""
    
    # Create directories
    Path("artifacts/features").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Save V1
    v1_artifact = "artifacts/features/athletes_features_v1.parquet"
    v1_feast = "data/athletes_features_v1.parquet"
    
    df_v1.to_parquet(v1_artifact, index=False)
    df_v1.to_parquet(v1_feast, index=False)
    
    # Save V2
    v2_artifact = "artifacts/features/athletes_features_v2.parquet"
    v2_feast = "data/athletes_features_v2.parquet"
    
    df_v2.to_parquet(v2_artifact, index=False)
    df_v2.to_parquet(v2_feast, index=False)
    
    logger.info(f"Feature versions saved:")
    logger.info(f"  V1: {v1_artifact}")
    logger.info(f"  V2: {v2_artifact}")
    
    return v1_artifact, v1_feast, v2_artifact, v2_feast

def main():
    """
    Main feature engineering pipeline
    """
    logger.info("Starting Feature Engineering Pipeline...")
    
    try:
        # Load clean data
        df_clean = load_clean_data()
        
        # Create two different feature versions
        df_v1 = create_feature_version_1(df_clean)
        df_v2 = create_feature_version_2(df_clean)
        
        # Save both versions
        v1_artifact, v1_feast, v2_artifact, v2_feast = save_feature_versions(df_v1, df_v2)
        
        # Summary report
        print(f"\n" + "="*60)
        print(f"FEATURE ENGINEERING SUMMARY")
        print(f"="*60)
        print(f"Base clean data: {df_clean.shape[0]} records, {df_clean.shape[1]} columns")
        print(f"")
        print(f"Version 1 (Basic Features):")
        print(f"  - Records: {df_v1.shape[0]}")
        print(f"  - Features: {df_v1.shape[1]}")
        print(f"  - Feature list: {list(df_v1.columns)}")
        print(f"")
        print(f"Version 2 (Engineered Features):")
        print(f"  - Records: {df_v2.shape[0]}")
        print(f"  - Features: {df_v2.shape[1]}")
        print(f"  - Additional features: {df_v2.shape[1] - df_v1.shape[1]}")
        print(f"  - New features: {list(set(df_v2.columns) - set(df_v1.columns))}")
        print(f"")
        print(f"Files created:")
        print(f"  - V1: {v1_feast}")
        print(f"  - V2: {v2_feast}")
        print(f"="*60)
        
        return df_v1, df_v2
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    features_v1, features_v2 = main()