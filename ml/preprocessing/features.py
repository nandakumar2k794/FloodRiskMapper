"""
Feature Matrix Preprocessing and Builder.
Integrates Tiers A-F and non-linear interactions.
"""
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from ml.feature_matrix import (
    MODEL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    DRAINWATCH_FEATURE_NAMES,
    FeatureMatrixBuilder,
)

__all__ = [
    "MODEL_FEATURE_NAMES",
    "CATEGORICAL_FEATURES",
    "DRAINWATCH_FEATURE_NAMES",
    "FeatureMatrixBuilder",
]
