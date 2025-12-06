# agents/data_ingestion_agent.py

import pandas as pd
import os

class DataIngestionAgent:
    def validate(self, files):
        # Validate product, sales, inventory, pricing data
        # For now, return files as-is if they exist, otherwise return empty dict
        validated = {}
        for key, filepath in files.items():
            if isinstance(filepath, str) and os.path.exists(filepath):
                validated[key] = filepath
            else:
                # Create dummy data if file doesn't exist
                validated[key] = None
        return validated

    def structure(self, raw_data):
        # Standardize to DataFrames or JSON
        structured = {}
        for key, filepath in raw_data.items():
            if filepath and os.path.exists(filepath):
                try:
                    structured[key] = pd.read_csv(filepath)
                except Exception:
                    # If CSV read fails, create empty DataFrame
                    structured[key] = pd.DataFrame()
            else:
                # Create dummy DataFrame for missing files
                structured[key] = pd.DataFrame()
        return structured

    def feature_engineering(self, df):
        # Compute velocity, avg price, etc.
        # For now, return the structured data as-is
        # Expected format: {'sales': DataFrame, 'inventory': DataFrame, 'price': DataFrame}
        return df

    def run(self, files):
        validated = self.validate(files)
        structured = self.structure(validated)
        features = self.feature_engineering(structured)
        # Ensure we return a dict with expected keys
        result = {
            'sales': features.get('sales', pd.DataFrame()),
            'inventory': features.get('inventory', pd.DataFrame()),
            'price': features.get('price', pd.DataFrame())
        }
        return result