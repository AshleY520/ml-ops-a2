"""
Training script that uses Feature Store for feature retrieval
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from codecarbon import EmissionsTracker
import logging
import os
from pathlib import Path
from feast import FeatureStore
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_features_from_store(feature_version):
    """Load features from the feature store"""
    try:
        store = FeatureStore(repo_path="feature_repo")
        logger.info(f"Connected to Feature Store")
        
        if feature_version == "v1":
            df = pd.read_parquet("data/athletes_features_v1.parquet")
        elif feature_version == "v2":
            df = pd.read_parquet("data/athletes_features_v2.parquet")
        else:
            raise ValueError(f"Unknown feature version: {feature_version}")
        
        logger.info(f"Loaded {feature_version} features: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error: {e}")
        # Fallback to direct loading
        if feature_version == "v1":
            df = pd.read_parquet("data/athletes_features_v1.parquet")
        else:
            df = pd.read_parquet("data/athletes_features_v2.parquet")
        return df

def get_feature_columns(feature_version):
    """Get feature columns based on version"""
    if feature_version == "v1":
        return ['age', 'weight', 'height', 'gender_encoded', 
                'deadlift', 'candj', 'snatch', 'backsq']
    else:  # v2
        return ['age', 'weight', 'height', 'gender_encoded',
                'deadlift', 'candj', 'snatch', 'backsq',
                'bmi', 'height_weight_ratio', 'strength_to_weight_ratio',
                'deadlift_to_weight_ratio', 'squat_to_deadlift_ratio',
                'olympic_lifts_total', 'powerlifting_total', 'olympic_to_powerlifting_ratio',
                'deadlift_per_kg', 'snatch_per_kg', 'candj_per_kg',
                'experience_numeric', 'training_frequency',
                'age_category_encoded', 'weight_category_encoded']

def prepare_features(df, feature_version):
    """Prepare features for training"""
    target_col = 'total_lift'
    feature_cols = get_feature_columns(feature_version)
    
    # Use only available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    
    logger.info(f"Using {len(available_cols)} features: {available_cols}")
    
    X = df[available_cols].fillna(df[available_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    return X, y, available_cols

def train_model(feature_version, n_estimators, max_depth):
    """Train model using features from Feature Store"""
    
    # Create directories
    Path("artifacts/carbon").mkdir(parents=True, exist_ok=True)
    
    # Start carbon tracking
    tracker = EmissionsTracker(
        project_name=f"athletes_ml_{feature_version}",
        output_dir="artifacts/carbon",
        save_to_file=True,
        save_to_api=False,
        log_level="error",
        output_file=f"emissions_{feature_version}_{n_estimators}_{max_depth}.csv"
    )
    tracker.start()
    
    try:
        mlflow_run_id = os.environ.get('MLFLOW_RUN_ID')
        
        if mlflow_run_id:
            logger.info(f"Running in MLflow pipeline context (Run ID: {mlflow_run_id})")
            
            unique_suffix = f"_{feature_version}_n{n_estimators}_d{max_depth}"
            mlflow.log_params({
                f"feature_version{unique_suffix}": feature_version,
                f"n_estimators{unique_suffix}": n_estimators,
                f"max_depth{unique_suffix}": max_depth,
                f"algorithm{unique_suffix}": "RandomForest"
            })
            
        else:
            mlflow.start_run(run_name=f"RF_{feature_version}_n{n_estimators}_d{max_depth}")
            logger.info(f"Created new MLflow run: {mlflow.active_run().info.run_id}")
            
            mlflow.log_params({
                "feature_version": feature_version,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "algorithm": "RandomForest"
            })
        
        df = load_features_from_store(feature_version)
        X, y, available_cols = prepare_features(df, feature_version)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training Random Forest: {n_estimators} trees, depth {max_depth}")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions and metrics
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        # Stop carbon tracking
        emissions = tracker.stop()
        
        # Log metrics with unique names in pipeline mode
        if mlflow_run_id:
            unique_suffix = f"_{feature_version}_n{n_estimators}_d{max_depth}"
            mlflow.log_metrics({
                f"train_r2{unique_suffix}": train_r2,
                f"test_r2{unique_suffix}": test_r2,
                f"train_mse{unique_suffix}": train_mse,
                f"test_mse{unique_suffix}": test_mse,
                f"carbon_emissions_kg{unique_suffix}": emissions,
                f"features_count{unique_suffix}": len(available_cols)
            })
        else:
            mlflow.log_metrics({
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "carbon_emissions_kg": emissions,
                "features_count": len(available_cols)
            })
        
        # Log artifacts
        mlflow.log_dict({"feature_names": available_cols}, f"features_{feature_version}_{n_estimators}_{max_depth}.json")
        
        # Print results
        logger.info("="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)
        logger.info(f"Feature Store Version: {feature_version}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Test MSE: {test_mse:.4f}")
        logger.info(f"Features Used: {len(available_cols)}")
        logger.info(f"Carbon Emissions: {emissions:.6f} kg CO₂")
        logger.info("="*60)
        
        if not mlflow_run_id:
            mlflow.end_run()
        
        return {
            'test_r2': test_r2,
            'test_mse': test_mse,
            'carbon_emissions': emissions,
            'feature_version': feature_version,
            'n_features': len(available_cols)
        }
        
    except Exception as e:
        tracker.stop()
        if not os.environ.get('MLFLOW_RUN_ID') and mlflow.active_run():
            mlflow.end_run()
        logger.error(f"Training failed: {e}")
        raise

def setup_mlflow_environment():
    import os
    from pathlib import Path
    
    Path("mlruns").mkdir(exist_ok=True)
    Path("artifacts/carbon").mkdir(parents=True, exist_ok=True)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    Path("artifacts/plots").mkdir(parents=True, exist_ok=True)
    
    mlflow.set_tracking_uri("file:./mlruns")
    
    experiment_name = "Athletes_Feature_Store_Pipeline"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"Using experiment: {experiment_name} (ID: {experiment.experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return True
        
    except Exception as e:
        print(f"Experiment setup warning: {e}")
        try:
            mlflow.set_experiment("Default")
            return True
        except:
            return False
        
def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    
    args = parser.parse_args()
    
    if not os.environ.get('MLFLOW_RUN_ID'):
        setup_mlflow_environment()
    else:
        print("Running in MLflow pipeline context")
    
    # Train model
    result = train_model(args.feature_version, args.n_estimators, args.max_depth)
    
    print(f"\n Training completed!")
    print(f"Feature Version: {result['feature_version']}")
    print(f"Test R²: {result['test_r2']:.4f}")
    print(f"Features: {result['n_features']}")

if __name__ == "__main__":
    main()