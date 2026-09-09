"""
predict.py - Standalone Inference Script for Chennai Flood Risk Model

Usage Examples:
  1. CLI single prediction:
     python predict.py --rainfall 180 --distance_to_road 20 --distance_to_drainage 50 --distance_to_river 300 --mean_elevation 6.5 --slope 0.8
  2. Batch CSV prediction:
     python predict.py --input-csv sample_features.csv --output-csv predictions.csv
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent

# Auto-locate model and feature files (local folder or repo structure)
def resolve_file(filename, repo_subpath):
    local_path = SCRIPT_DIR / filename
    if local_path.exists():
        return local_path
    repo_path = SCRIPT_DIR.parent / repo_subpath / filename
    if repo_path.exists():
        return repo_path
    return local_path

MODEL_PATH = resolve_file("final_model.json", "models/xgboost")
FEATURES_PATH = resolve_file("feature_columns.json", "models/xgboost")

def classify_risk(p):
    if pd.isna(p):
        return "Unknown"
    if p <= 0.25:
        return "Low"
    if p <= 0.50:
        return "Moderate"
    if p <= 0.75:
        return "High"
    return "Very High"

class FloodPredictor:
    def __init__(self, model_path=MODEL_PATH, features_path=FEATURES_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}.")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature columns definition not found at {features_path}.")
            
        with open(features_path, "r") as f:
            self.feature_columns = json.load(f)
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))

    def predict(self, data):
        """
        Accepts:
          - dict of features (single sample)
          - pandas DataFrame (batch samples)
        Returns:
          - dict with flood_probability & risk_class OR DataFrame with predictions
        """
        if isinstance(data, dict):
            row = {}
            for col in self.feature_columns:
                row[col] = [data.get(col, np.nan)]
            df = pd.DataFrame(row)
            prob = float(self.model.predict_proba(df)[:, 1][0])
            risk = classify_risk(prob)
            return {
                "flood_probability": round(prob, 4),
                "risk_class": risk
            }
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = np.nan
            probs = self.model.predict_proba(df[self.feature_columns])[:, 1]
            df["flood_probability"] = np.round(probs, 4)
            df["risk_class"] = [classify_risk(p) for p in probs]
            return df
        else:
            raise TypeError("Data must be a dictionary or a pandas DataFrame.")

def main():
    parser = argparse.ArgumentParser(description="Chennai Flood Risk Inference")
    parser.add_argument("--input-csv", type=str, help="Path to input feature CSV for batch inference")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Path to save output CSV")
    
    # Feature arguments for single prediction
    parser.add_argument("--rainfall", type=float, default=150.0, help="Annual/event rainfall (mm)")
    parser.add_argument("--distance_to_road", type=float, default=50.0, help="Distance to road (m)")
    parser.add_argument("--distance_to_drainage", type=float, default=100.0, help="Distance to stormwater drain (m)")
    parser.add_argument("--distance_to_river", type=float, default=500.0, help="Distance to river (m)")
    parser.add_argument("--drainage_density", type=float, default=10.0, help="Drainage density index")
    parser.add_argument("--land_use", type=int, default=1, help="Land use category code")
    parser.add_argument("--soil", type=int, default=1, help="Soil category code")
    parser.add_argument("--mean_elevation", type=float, default=6.0, help="Mean elevation (m)")
    parser.add_argument("--min_elevation", type=float, default=5.0, help="Min elevation (m)")
    parser.add_argument("--max_elevation", type=float, default=7.0, help="Max elevation (m)")
    parser.add_argument("--elevation_range", type=float, default=2.0, help="Elevation range (m)")
    parser.add_argument("--elevation_std", type=float, default=0.5, help="Elevation standard deviation (m)")
    parser.add_argument("--slope", type=float, default=0.8, help="Slope (degrees)")

    args = parser.parse_args()
    predictor = FloodPredictor()

    if args.input_csv:
        input_path = Path(args.input_csv)
        if not input_path.exists():
            print(f"Error: {input_path} not found.")
            sys.exit(1)
        df_in = pd.read_csv(input_path)
        print(f"Running inference on {len(df_in)} samples from {input_path}...")
        df_out = predictor.predict(df_in)
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"Inference completed! Output saved to: {out_path}")
        print("\nRisk class distribution:")
        print(df_out['risk_class'].value_counts().to_string())
    else:
        sample = vars(args)
        sample.pop("input_csv")
        sample.pop("output_csv")
        result = predictor.predict(sample)
        print("\n" + "="*40)
        print("SINGLE GRID-CELL INFERENCE RESULT")
        print("="*40)
        print(f"Flood Probability: {result['flood_probability']:.4f}")
        print(f"Risk Class:        {result['risk_class']}")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()
