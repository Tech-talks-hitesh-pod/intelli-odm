# agents/data_ingestion_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

# Try to import Pydantic for validation
try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None
    field_validator = None
    model_validator = None
    ValidationError = None

class DataIngestionAgent:
    """Enhanced Data Ingestion Agent with Pydantic validation and LLM integration"""
    
    def __init__(self, ollama_client=None, audit_logger=None):
        self.ollama_client = ollama_client
        self.audit_logger = audit_logger
        
        # Define expected schemas for each data type
        self.schemas = {
            'sales_data': {
                'required_columns': ['date', 'store_id', 'sku', 'units_sold', 'revenue'],
                'optional_columns': ['city', 'avgdiscount', 'pricebucket'],
                'description': 'Historical sales data with date, store, SKU, units sold, and revenue'
            },
            'inventory_data': {
                'required_columns': ['store_id', 'sku', 'on_hand'],
                'optional_columns': ['in_transit'],
                'description': 'Current inventory levels by store and SKU'
            },
            'price_data': {
                'required_columns': ['store_id', 'sku', 'price'],
                'optional_columns': ['markdown_flag'],
                'description': 'Pricing information by store and SKU'
            },
            'cost_data': {
                'required_columns': ['sku', 'cost'],
                'optional_columns': ['currency'],
                'description': 'Cost data for each SKU'
            },
            'new_articles_data': {
                'required_columns': ['product_id', 'vendor_sku', 'description'],
                'optional_columns': ['category', 'color', 'material', 'size_set', 'brick', 'class', 'segment', 'family', 'brand'],
                'description': 'New product articles with attributes'
            }
        }
    
    def validate_with_llm(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Use LLM to validate data quality and provide insights"""
        if not self.ollama_client:
            return {'valid': True, 'warnings': [], 'llm_validation': False}
        
        schema = self.schemas.get(data_type, {})
        required_cols = schema.get('required_columns', [])
        
        # Check basic structure
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'valid': False,
                'errors': [f"Missing required columns: {', '.join(missing_cols)}"],
                'warnings': []
            }
        
        # Prepare data summary for LLM
        data_summary = {
            'row_count': len(df),
            'columns': list(df.columns),
            'sample_rows': df.head(3).to_dict('records') if len(df) > 0 else [],
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Create validation prompt
        prompt = f"""You are a data quality validator. Analyze this {data_type} dataset:

Schema Requirements:
- Required columns: {', '.join(required_cols)}
- Optional columns: {', '.join(schema.get('optional_columns', []))} (these are optional and can have empty values)
- Description: {schema.get('description', '')}

Data Summary:
{json.dumps(data_summary, indent=2, default=str)}

IMPORTANT VALIDATION RULES:
1. Optional columns are allowed to have empty/null values. Only required columns must have non-empty values in all rows.
2. The 'size_set' column is EXPECTED to contain comma-separated values (e.g., "S,M,L,XL" or "30,32,34,36"). This is NOT an error or warning.
3. Comma-separated values in any column are acceptable if they represent valid data (like size ranges, multiple options, etc.).

Please validate:
1. Are all required columns present and do they have non-empty values?
2. Are there any critical data quality issues (missing required data, wrong types for required fields)?
3. Are there any anomalies or inconsistencies that would prevent data processing?
4. Provide recommendations for data improvement.

DO NOT flag as warnings:
- Comma-separated values in columns (especially size_set)
- Empty values in optional columns
- String data types (object type) - this is normal for text data

Respond in JSON format:
{{
    "valid": true/false,
    "errors": ["list of critical errors only - missing required columns or invalid required data"],
    "warnings": ["list of warnings - only serious data quality concerns that don't prevent processing"],
    "recommendations": ["list of recommendations"],
    "data_quality_score": 0-100
}}"""
        
        try:
            response = self.ollama_client.generate(prompt, max_tokens=1000)
            # Try to parse JSON from response
            if isinstance(response, str):
                # Extract JSON from response if it's wrapped in markdown
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        validation_result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # Fallback: create result from text
                        validation_result = {
                            'valid': 'error' not in response.lower() and 'invalid' not in response.lower(),
                            'errors': [],
                            'warnings': [response[:200]] if len(response) > 0 else [],
                            'recommendations': [],
                            'data_quality_score': 70
                        }
                else:
                    # Fallback: create result from text
                    validation_result = {
                        'valid': 'error' not in response.lower() and 'invalid' not in response.lower(),
                        'errors': [],
                        'warnings': [response[:200]] if len(response) > 0 else [],
                        'recommendations': [],
                        'data_quality_score': 70
                    }
            else:
                validation_result = response if isinstance(response, dict) else {
                    'valid': True,
                    'errors': [],
                    'warnings': [],
                    'recommendations': [],
                    'data_quality_score': 80
                }
            
            return {
                'valid': validation_result.get('valid', True),
                'errors': validation_result.get('errors', []),
                'warnings': validation_result.get('warnings', []),
                'recommendations': validation_result.get('recommendations', []),
                'data_quality_score': validation_result.get('data_quality_score', 100),
                'llm_validation': True
            }
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_agent_operation(
                    agent_name="DataIngestionAgent",
                    description=f"LLM validation failed: {str(e)}",
                    status="WARNING"
                )
            return {
                'valid': True,
                'warnings': [f"LLM validation unavailable: {str(e)}"],
                'llm_validation': False
            }
    
    def validate(self, file_path: str, data_type: str) -> Dict[str, Any]:
        """Validate data file structure and content"""
        try:
            # Read file - keep empty strings as empty strings, don't convert to NaN
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, keep_default_na=False, na_values=[''])
            else:
                return {
                    'valid': False,
                    'errors': [f"Unsupported file format: {Path(file_path).suffix}"],
                    'warnings': []
                }
            
            # Replace any remaining NaN/None with empty strings for optional columns
            schema = self.schemas.get(data_type, {})
            optional_cols = schema.get('optional_columns', [])
            for col in optional_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            
            # Basic validation
            schema = self.schemas.get(data_type, {})
            required_cols = schema.get('required_columns', [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            validation_result = {
                'valid': len(missing_cols) == 0,
                'errors': [f"Missing required columns: {', '.join(missing_cols)}"] if missing_cols else [],
                'warnings': [],
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            
            # Additional checks
            if len(df) == 0:
                validation_result['warnings'].append("Dataset is empty")
                validation_result['valid'] = False
            
            # Check for excessive nulls
            null_percentages = df.isnull().sum() / len(df) * 100
            high_null_cols = null_percentages[null_percentages > 50].index.tolist()
            if high_null_cols:
                validation_result['warnings'].append(
                    f"Columns with >50% nulls: {', '.join(high_null_cols)}"
                )
            
            # LLM validation if available
            if self.ollama_client:
                llm_result = self.validate_with_llm(df, data_type)
                validation_result.update(llm_result)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def structure(self, file_path: str, data_type: str) -> pd.DataFrame:
        """Standardize data format"""
        try:
            # Read file - keep empty strings as empty strings
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, keep_default_na=False, na_values=[''])
            else:
                raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")
            
            # Standardize column names (lowercase, strip spaces)
            df.columns = df.columns.str.lower().str.strip()
            
            # Replace NaN/None with empty strings for optional columns
            schema = self.schemas.get(data_type, {})
            optional_cols = schema.get('optional_columns', [])
            for col in optional_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('')
            
            # Convert date columns if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error structuring data: {str(e)}")
    
    def feature_engineering(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Compute derived features"""
        features = {}
        
        if data_type == 'sales_data':
            if 'units_sold' in df.columns and 'date' in df.columns:
                # Sales velocity
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df_sorted = df.sort_values('date')
                if len(df_sorted) > 1:
                    days_diff = (df_sorted['date'].max() - df_sorted['date'].min()).days
                    if days_diff > 0:
                        total_units = df['units_sold'].sum()
                        features['sales_velocity'] = total_units / days_diff
                
                # Average price
                if 'revenue' in df.columns and 'units_sold' in df.columns:
                    features['avg_price'] = (df['revenue'] / df['units_sold']).mean()
            
        elif data_type == 'inventory_data':
            if 'on_hand' in df.columns:
                features['total_inventory'] = df['on_hand'].sum()
                features['avg_inventory_per_sku'] = df['on_hand'].mean()
        
        elif data_type == 'price_data':
            if 'price' in df.columns:
                features['avg_price'] = df['price'].mean()
                features['price_range'] = [df['price'].min(), df['price'].max()]
        
        features['row_count'] = len(df)
        features['column_count'] = len(df.columns)
        
        return features
    
    def run(self, file_path: str, data_type: str) -> Dict[str, Any]:
        """Complete data ingestion pipeline"""
        # Validate
        validation = self.validate(file_path, data_type)
        
        if not validation['valid']:
            return {
                'success': False,
                'validation': validation,
                'data': None,
                'features': {}
            }
        
        # Structure
        try:
            df = self.structure(file_path, data_type)
        except Exception as e:
            return {
                'success': False,
                'validation': {
                    **validation,
                    'errors': validation.get('errors', []) + [str(e)]
                },
                'data': None,
                'features': {}
            }
        
        # Feature engineering
        features = self.feature_engineering(df, data_type)
        
        return {
            'success': True,
            'validation': validation,
            'data': df,
            'features': features,
            'file_path': file_path
        }
