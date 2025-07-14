"""
Athlete Features Version 2 - Engineered Features
"""

from feast import FeatureView, Field, FileSource
from feast.types import Float64, String, Int64
from datetime import timedelta
from .entities import athlete

# Data source for V2
athlete_source_v2 = FileSource(
    path="../data/athletes_features_v2.parquet", 
    timestamp_field="event_timestamp", 
    created_timestamp_column="created_timestamp",
)

# Feature view v2 - Enhanced features
athlete_features_v2 = FeatureView(
    name="athlete_features_v2",
    entities=[athlete],
    schema=[
        # Basic features (same as V1)
        Field(name="age", dtype=Float64),
        Field(name="weight", dtype=Float64),
        Field(name="height", dtype=Float64),
        Field(name="gender_encoded", dtype=Int64),
        Field(name="deadlift", dtype=Float64),
        Field(name="candj", dtype=Float64),
        Field(name="snatch", dtype=Float64),
        Field(name="backsq", dtype=Float64),
        Field(name="total_lift", dtype=Float64),
        Field(name="experience", dtype=String),
        Field(name="schedule", dtype=String),
        
        # Engineered features (V2 only)
        Field(name="bmi", dtype=Float64),
        Field(name="height_weight_ratio", dtype=Float64),
        Field(name="strength_to_weight_ratio", dtype=Float64),
        Field(name="deadlift_to_weight_ratio", dtype=Float64),
        Field(name="squat_to_deadlift_ratio", dtype=Float64),
        Field(name="olympic_lifts_total", dtype=Float64),
        Field(name="powerlifting_total", dtype=Float64),
        Field(name="olympic_to_powerlifting_ratio", dtype=Float64),
        Field(name="deadlift_per_kg", dtype=Float64),
        Field(name="snatch_per_kg", dtype=Float64),
        Field(name="candj_per_kg", dtype=Float64),
        Field(name="experience_numeric", dtype=Int64),
        Field(name="training_frequency", dtype=Int64),
        Field(name="age_category_encoded", dtype=Int64),
        Field(name="weight_category_encoded", dtype=Int64),
    ],
    source=athlete_source_v2,
    ttl=timedelta(days=365),
    description="Advanced athlete features with engineering (Version 2)"
)