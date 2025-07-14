import pandas as pd
import os
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("data", exist_ok=True)

def clean_raw_data(df):
    """
    Comprehensive data cleaning based on project requirements
    """
    logger.info(f"Starting data cleaning with {len(df)} records")
    
    # Remove not relevant columns first
    columns_to_drop = ['affiliate', 'team', 'name', 'fran', 'helen', 'grace',
                      'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'train']
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    logger.info(f"Dropped {len(columns_to_drop)} irrelevant columns")
    
    # Define required columns
    required_cols = ['region', 'age', 'weight', 'height', 'gender', 'eat',
                    'background', 'experience', 'schedule', 'howlong',
                    'deadlift', 'candj', 'snatch', 'backsq']
    
    # Clean survey data - handle decline to answer
    decline_dict = {'Decline to answer|': np.nan, 'Decline to answer': np.nan}
    df = df.replace(decline_dict)
    
    # Drop rows with missing critical data
    df = df.dropna(subset=required_cols)
    logger.info(f"After removing rows with missing critical data: {len(df)} records")
    
    # Convert numeric columns
    numeric_cols = ["deadlift", "candj", "snatch", "backsq", "age", "weight", "height"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove outliers - basic filters
    df = df[df['weight'] < 1500]  # Unrealistic weight
    df = df[df['gender'] != '--']  # Invalid gender
    df = df[df['age'] >= 18]  # Adults only
    df = df[(df['height'] < 96) & (df['height'] > 48)]  # Reasonable height range (inches)
    
    logger.info(f"After basic outlier removal: {len(df)} records")
    
    # Remove outliers for lifting weights by gender
    # Male deadlift: 0 < deadlift <= 1105
    # Female deadlift: 0 < deadlift <= 636
    male_mask = (df['gender'] == 'Male')
    female_mask = (df['gender'] == 'Female')
    
    valid_deadlift = (
        (male_mask & (df['deadlift'] > 0) & (df['deadlift'] <= 1105)) |
        (female_mask & (df['deadlift'] > 0) & (df['deadlift'] <= 636))
    )
    df = df[valid_deadlift]
    
    # Other lift constraints
    df = df[(df['candj'] > 0) & (df['candj'] <= 395)]
    df = df[(df['snatch'] > 0) & (df['snatch'] <= 496)]
    df = df[(df['backsq'] > 0) & (df['backsq'] <= 1069)]
    
    logger.info(f"After lifting weight outlier removal: {len(df)} records")
    
    # Calculate total lift (basic feature)
    lift_cols = ["deadlift", "candj", "snatch", "backsq"]
    df["total_lift"] = df[lift_cols].sum(axis=1)
    
    logger.info(f"Final cleaned data has {len(df)} records")
    return df

def add_required_metadata(df):
    """
    Add minimal required metadata for feature store
    """
    # Add unique athlete identifier
    df = df.reset_index(drop=True)
    df["athlete_id"] = df.index.astype(str)
    
    # Add timestamps required by Feast
    df["event_timestamp"] = pd.Timestamp.now(tz='UTC')
    df["created_timestamp"] = pd.Timestamp.now(tz='UTC')
    
    return df

def save_clean_data(df):
    """
    Save the cleaned base dataset
    """
    # Save as parquet for better performance
    base_file = "artifacts/athletes_clean.parquet"
    df.to_parquet(base_file, index=False)
    logger.info(f"Saved clean base data to {base_file}")
    
    # Also save to data directory for feature store access
    feast_file = "data/athletes_clean.parquet"  
    df.to_parquet(feast_file, index=False)
    logger.info(f"Saved clean data for feature store to {feast_file}")
    
    return base_file, feast_file

def main():
    """
    Simple data ingestion and cleaning pipeline
    """
    logger.info("Starting simple data ingestion pipeline...")
    
    try:
        # Load raw data
        df_raw = pd.read_csv("data/athletes.csv")
        logger.info(f"Loaded raw data: {df_raw.shape}")
        
        # Basic cleaning only
        df_clean = clean_raw_data(df_raw)
        
        # Add required metadata
        df_final = add_required_metadata(df_clean)
        
        # Save cleaned data
        base_file, feast_file = save_clean_data(df_final)
        
        # Summary
        logger.info("Data ingestion completed successfully!")
        logger.info(f"Raw records: {len(df_raw)}")
        logger.info(f"Clean records: {len(df_final)}")
        logger.info(f"Columns: {list(df_final.columns)}")
        
        # Basic stats
        print(f"\nData Summary:")
        print(f"- Records: {len(df_final)}")
        print(f"- Features: {len(df_final.columns)}")
        print(f"- Gender distribution: {df_final['gender'].value_counts().to_dict()}")
        print(f"- Age range: {df_final['age'].min():.0f} - {df_final['age'].max():.0f}")
        print(f"- Clean data saved to: {base_file}")
        
        return df_final
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    processed_data = main()