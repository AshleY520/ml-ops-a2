"""
Athlete Features Version 1 - Basic Features
"""

from feast import FeatureView, Field, FileSource
from feast.types import Float64, String, Int64
from datetime import timedelta
from .entities import athlete

# Data source for V1
athlete_source_v1 = FileSource(
    path="../data/athletes_features_v1.parquet", 
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature view v1 - Basic features
athlete_features_v1 = FeatureView(
    name="athlete_features_v1", 
    entities=[athlete],
    schema=[
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
    ],
    source=athlete_source_v1,
    ttl=timedelta(days=365),
    description="Basic athlete features - minimal engineering (Version 1)"
)