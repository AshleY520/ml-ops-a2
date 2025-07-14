"""
Entity definitions for the athlete feature store
"""

from feast import Entity
from feast import ValueType

athlete = Entity(
    name="athlete",
    description="Athlete identifier",
    value_type=ValueType.INT64  
)