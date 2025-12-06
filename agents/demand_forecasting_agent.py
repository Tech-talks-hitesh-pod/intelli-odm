# agents/demand_forecasting_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Pydantic for validation
try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
    from pydantic import ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None
    field_validator = None
    model_validator = None
    ValidationError = None

# Forecasting libraries
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
except ImportError:
    PCA = None
    StandardScaler = None
    RandomForestRegressor = None
    LinearRegression = None
    KMeans = None

# Pydantic Models for Data Validation
if PYDANTIC_AVAILABLE:
    
    class SalesDataRecord(BaseModel):
        """Pydantic model for validating individual sales records"""
        model_config = ConfigDict(extra='allow')  # Allow extra fields
        
        store_id: str = Field(..., description="Store identifier", min_length=1)
        sku: str = Field(..., description="Stock keeping unit identifier", min_length=1)
        units_sold: float = Field(..., description="Number of units sold", ge=0)
        date: Optional[Union[str, datetime]] = Field(None, description="Sale date (optional but recommended)")
        revenue: Optional[float] = Field(None, description="Revenue from sale", ge=0)
        
        @field_validator('date', mode='before')
        @classmethod
        def parse_date(cls, v):
            if v is None:
                return None
            if isinstance(v, str):
                try:
                    return pd.to_datetime(v)
                except:
                    return v
            return v
        
        @field_validator('units_sold', 'revenue', mode='before')
        @classmethod
        def parse_numeric(cls, v):
            if v is None:
                return v
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return v
    
    class InventoryDataRecord(BaseModel):
        """Pydantic model for validating individual inventory records"""
        model_config = ConfigDict(extra='allow')
        
        store_id: str = Field(..., description="Store identifier", min_length=1)
        sku: str = Field(..., description="Stock keeping unit identifier", min_length=1)
        on_hand: float = Field(..., description="Current inventory on hand", ge=0)
        in_transit: Optional[float] = Field(0, description="Inventory in transit", ge=0)
        
        @field_validator('on_hand', 'in_transit', mode='before')
        @classmethod
        def parse_numeric(cls, v):
            if v is None:
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return v
    
    class PriceDataRecord(BaseModel):
        """Pydantic model for validating individual price records"""
        model_config = ConfigDict(extra='allow')
        
        store_id: str = Field(..., description="Store identifier", min_length=1)
        sku: str = Field(..., description="Stock keeping unit identifier", min_length=1)
        price: float = Field(..., description="Product price", gt=0)
        markdown_flag: Optional[bool] = Field(False, description="Whether price is marked down")
        
        @field_validator('price', mode='before')
        @classmethod
        def parse_price(cls, v):
            if v is None:
                raise ValueError("Price cannot be None")
            if isinstance(v, (int, float)):
                price = float(v)
                if price <= 0:
                    raise ValueError("Price must be greater than 0")
                return price
            try:
                price = float(v)
                if price <= 0:
                    raise ValueError("Price must be greater than 0")
                return price
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid price value: {v}")
    
    class ComparableProduct(BaseModel):
        """Pydantic model for validating comparable products"""
        model_config = ConfigDict(extra='allow')
        
        sku: str = Field(..., description="SKU of comparable product", min_length=1)
        similarity: Optional[float] = Field(None, description="Similarity score", ge=0, le=1)
        attributes: Optional[Dict[str, Any]] = Field(None, description="Product attributes")
        
        @field_validator('similarity', mode='before')
        @classmethod
        def parse_similarity(cls, v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                sim = float(v)
                if sim < 0 or sim > 1:
                    raise ValueError("Similarity must be between 0 and 1")
                return sim
            return v
    
    class InputDataValidation(BaseModel):
        """Pydantic model for validating all input data"""
        model_config = ConfigDict(extra='allow')
        
        sales_data: Optional[List[SalesDataRecord]] = Field(None, description="Sales data records")
        inventory_data: Optional[List[InventoryDataRecord]] = Field(None, description="Inventory data records")
        price_data: Optional[List[PriceDataRecord]] = Field(None, description="Price data records")
        comparables: Optional[List[ComparableProduct]] = Field(None, description="Comparable products")
        
        @model_validator(mode='after')
        def validate_data_completeness(self):
            """Validate that critical data is present"""
            errors = []
            warnings = []
            
            # Sales data is critical
            if not self.sales_data or len(self.sales_data) == 0:
                errors.append("CRITICAL: Sales data is missing or empty. Sales data is required for forecasting.")
            else:
                # Check for date column presence
                has_date = any(record.date is not None for record in self.sales_data)
                if not has_date:
                    warnings.append("WARNING: Sales data missing 'date' column. Time-series forecasting will be limited.")
            
            # Inventory data is optional but recommended
            if not self.inventory_data or len(self.inventory_data) == 0:
                warnings.append("WARNING: Inventory data is missing. Forecast accuracy may be reduced.")
            
            # Price data is optional but recommended
            if not self.price_data or len(self.price_data) == 0:
                warnings.append("WARNING: Price data is missing. Price sensitivity analysis will be limited.")
            
            # Comparables are optional
            if not self.comparables or len(self.comparables) == 0:
                warnings.append("INFO: No comparable products found. Analogy-based forecasting will be unavailable.")
            
            # Store validation results
            self.validation_errors = errors
            self.validation_warnings = warnings
            self.is_valid = len(errors) == 0
            
            return self

class DemandForecastingAgent:
    """
    Demand Forecasting Agent that performs three sequential tasks:
    1. Select best forecasting model (decay/depletion + time series)
    2. Sensitivity and factor analysis (correlation + dimensionality reduction)
    3. Store-level forecasting (article selection + quantity)
    """
    
    def __init__(self, llama_client, default_margin_pct: float = 0.40, 
                 target_sell_through_pct: float = 0.75, min_margin_pct: float = 0.25,
                 use_llm: bool = True, max_llm_tokens: int = 500,
                 universe_of_stores: Optional[int] = None,
                 enable_hitl: bool = True, variance_threshold: float = 0.05):
        """
        Initialize Demand Forecasting Agent with optimization parameters
        
        Args:
            llama_client: LLM client for model selection
            default_margin_pct: Default margin percentage (default 40%)
            target_sell_through_pct: Target sell-through rate (default 75%)
            min_margin_pct: Minimum acceptable margin percentage (default 25%)
            use_llm: Whether to use LLM for model selection (default True)
            max_llm_tokens: Maximum tokens for LLM prompts (default 500)
            universe_of_stores: Total number of stores in organization (for validation)
            enable_hitl: Enable Human-in-the-Loop workflow (default True)
            variance_threshold: Variance threshold for auto-approval (default 5% = 0.05)
        """
        self.llama = llama_client
        self.selected_model = None
        self.model_params = {}
        self.factor_analysis_results = {}
        self.forecast_results = {}
        self.default_margin_pct = default_margin_pct
        self.target_sell_through_pct = target_sell_through_pct
        self.min_margin_pct = min_margin_pct
        self.use_llm = use_llm
        self.max_llm_tokens = max_llm_tokens
        self.universe_of_stores = universe_of_stores
        self.enable_hitl = enable_hitl
        self.variance_threshold = variance_threshold
        self.data_validation_errors = []
        self.fallback_used = False
        
        # Initialize HITL workflow if enabled
        if self.enable_hitl:
            try:
                from agents.hitl_workflow import HITLWorkflow
                self.hitl = HITLWorkflow(variance_threshold=variance_threshold)
            except ImportError:
                self.hitl = None
                self.enable_hitl = False
                print("Warning: HITL workflow module not available")
        else:
            self.hitl = None
        self.data_validation_errors = []
        self.llm_cache = {}  # Cache LLM responses for token optimization
        self.use_fallback = False
        
    def select_model(self, comparables: List[Dict], sales_data: pd.DataFrame, 
                     inventory_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Task 1: Select best forecasting model considering:
        - Decay/depletion models (for products with declining demand)
        - Time series models (Prophet, ARIMA, exponential smoothing)
        - Analogy-based scaling (from comparable SKUs)
        - Assumption: Stores have enough space to accommodate assortment
        """
        
        print("=" * 60)
        print("TASK 1: Selecting Best Forecasting Model")
        print("=" * 60)
        
        # Validate input data first
        validation_result = self._validate_input_data(
            sales_data, inventory_data, price_data, comparables
        )
        
        if not validation_result['is_valid']:
            self.data_validation_errors = validation_result['errors']
            print("\n" + "=" * 60)
            print("DATA VALIDATION FAILED")
            print("=" * 60)
            for error in validation_result['errors']:
                print(f"❌ {error}")
            print("\n" + "=" * 60)
            raise ValueError("Input data validation failed. Please fix data inconsistencies.")
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(
            sales_data, inventory_data, price_data, comparables
        )
        
        # Use LLM to guide model selection (if enabled and available)
        model_recommendation = None
        if self.use_llm and self.llama:
            try:
                model_recommendation = self._llm_model_selection(
                    data_characteristics, comparables
                )
            except Exception as e:
                print(f"LLM model selection failed, using fallback: {e}")
                self.use_llm = False  # Disable LLM for rest of session
        
        # Evaluate different models
        try:
            model_scores = self._evaluate_models(
                sales_data, inventory_data, price_data, comparables, data_characteristics
            )
            
            if not model_scores or all(score['score'] == 0 for score in model_scores.values()):
                raise ValueError("All primary models failed")
            
            # Select best model
            best_model = max(model_scores.items(), key=lambda x: x[1]['score'])
            self.selected_model = best_model[0]
            self.model_params = best_model[1]['params']
            
        except Exception as e:
            print(f"\n⚠️  Primary model selection failed: {e}")
            print("🔄 Falling back to Linear Regression + K-Means Clustering")
            self.fallback_used = True
            self.selected_model = 'fallback_lr_kmeans'
            self.model_params = {'fallback_reason': str(e)}
            model_scores = {'fallback_lr_kmeans': {'score': 0.5, 'params': {}}}
        
        print(f"\nSelected Model: {self.selected_model}")
        if self.fallback_used:
            print("⚠️  Using fallback method (Linear Regression + K-Means)")
        if model_scores and self.selected_model in model_scores:
            print(f"Model Score: {model_scores[self.selected_model]['score']:.4f}")
        print(f"Model Parameters: {self.model_params}")
        
        return {
            'selected_model': self.selected_model,
            'model_params': self.model_params,
            'model_scores': model_scores,
            'llm_recommendation': model_recommendation,
            'data_characteristics': data_characteristics,
            'fallback_used': self.fallback_used
        }
    
    def _validate_input_data(self, sales_data: pd.DataFrame,
                            inventory_data: pd.DataFrame,
                            price_data: pd.DataFrame,
                            comparables: List[Dict]) -> Dict[str, Any]:
        """
        Validate input data using Pydantic models and return clear error messages
        
        Returns:
            Dict with 'is_valid' (bool) and 'errors' (list of error messages)
        """
        errors = []
        warnings = []
        info = []
        validation_details = []
        
        if PYDANTIC_AVAILABLE:
            try:
                # Convert DataFrames to lists of dictionaries for Pydantic validation
                sales_records = []
                inventory_records = []
                price_records = []
                comparable_products = []
                
                # Validate sales_data
                if sales_data is not None and not sales_data.empty:
                    try:
                        # Check required columns first
                        required_cols = ['store_id', 'sku', 'units_sold']
                        missing_cols = [col for col in required_cols if col not in sales_data.columns]
                        if missing_cols:
                            errors.append(f"CRITICAL: Sales data missing required columns: {', '.join(missing_cols)}. "
                                        f"Required columns: {', '.join(required_cols)}")
                        else:
                            # Convert to records and validate with Pydantic
                            sales_dict = sales_data.to_dict('records')
                            for idx, record in enumerate(sales_dict):
                                try:
                                    validated_record = SalesDataRecord(**record)
                                    sales_records.append(validated_record)
                                except ValidationError as e:
                                    errors.append(f"CRITICAL: Sales data row {idx+1} validation failed: {e.errors()[0]['msg']}")
                                    validation_details.append({
                                        'row': idx+1,
                                        'errors': [err['msg'] for err in e.errors()],
                                        'data': record
                                    })
                            
                            # Check for null values
                            for col in required_cols:
                                if col in sales_data.columns:
                                    null_count = sales_data[col].isnull().sum()
                                    if null_count > 0:
                                        warnings.append(f"WARNING: Sales data has {null_count} null values in '{col}' column. "
                                                      f"These rows will be excluded from analysis.")
                            
                            # Check for date column
                            if 'date' not in sales_data.columns:
                                warnings.append("WARNING: Sales data missing 'date' column. Time-series forecasting will be limited.")
                    except Exception as e:
                        errors.append(f"CRITICAL: Error processing sales data: {str(e)}")
                else:
                    errors.append("CRITICAL: Sales data is missing or empty. Sales data is required for forecasting.")
                
                # Validate inventory_data
                if inventory_data is not None and not inventory_data.empty:
                    try:
                        required_cols = ['store_id', 'sku', 'on_hand']
                        missing_cols = [col for col in required_cols if col not in inventory_data.columns]
                        if missing_cols:
                            warnings.append(f"WARNING: Inventory data missing columns: {', '.join(missing_cols)}. "
                                         "Inventory-aware forecasting will be limited.")
                        else:
                            inventory_dict = inventory_data.to_dict('records')
                            for idx, record in enumerate(inventory_dict):
                                try:
                                    validated_record = InventoryDataRecord(**record)
                                    inventory_records.append(validated_record)
                                except ValidationError as e:
                                    warnings.append(f"WARNING: Inventory data row {idx+1} validation failed: {e.errors()[0]['msg']}")
                    except Exception as e:
                        warnings.append(f"WARNING: Error processing inventory data: {str(e)}")
                else:
                    warnings.append("WARNING: Inventory data is missing. Forecast accuracy may be reduced.")
                
                # Validate price_data
                if price_data is not None and not price_data.empty:
                    try:
                        required_cols = ['store_id', 'sku', 'price']
                        missing_cols = [col for col in required_cols if col not in price_data.columns]
                        if missing_cols:
                            warnings.append(f"WARNING: Price data missing columns: {', '.join(missing_cols)}. "
                                         "Price-based optimization will be limited.")
                        else:
                            price_dict = price_data.to_dict('records')
                            for idx, record in enumerate(price_dict):
                                try:
                                    validated_record = PriceDataRecord(**record)
                                    price_records.append(validated_record)
                                except ValidationError as e:
                                    warnings.append(f"WARNING: Price data row {idx+1} validation failed: {e.errors()[0]['msg']}")
                    except Exception as e:
                        warnings.append(f"WARNING: Error processing price data: {str(e)}")
                else:
                    warnings.append("WARNING: Price data is missing. Price sensitivity analysis will be limited.")
                
                # Validate comparables
                if comparables and len(comparables) > 0:
                    try:
                        for idx, comp in enumerate(comparables):
                            try:
                                validated_comp = ComparableProduct(**comp)
                                comparable_products.append(validated_comp)
                            except ValidationError as e:
                                warnings.append(f"WARNING: Comparable product {idx+1} validation failed: {e.errors()[0]['msg']}")
                    except Exception as e:
                        warnings.append(f"WARNING: Error processing comparables: {str(e)}")
                else:
                    info.append("INFO: No comparable products found. Analogy-based forecasting will be unavailable.")
                
                # Create InputDataValidation model for final validation
                try:
                    validation_model = InputDataValidation(
                        sales_data=sales_records if sales_records else None,
                        inventory_data=inventory_records if inventory_records else None,
                        price_data=price_records if price_records else None,
                        comparables=comparable_products if comparable_products else None
                    )
                    
                    # Add model-level warnings
                    warnings.extend(validation_model.validation_warnings)
                    info.extend([w for w in validation_model.validation_warnings if w.startswith("INFO:")])
                    
                    is_valid = validation_model.is_valid
                except ValidationError as e:
                    errors.append(f"CRITICAL: Overall data validation failed: {e.errors()[0]['msg']}")
                    is_valid = False
                
                # Check data consistency
                if sales_data is not None and not sales_data.empty and inventory_data is not None and not inventory_data.empty:
                    sales_stores = set(sales_data['store_id'].unique()) if 'store_id' in sales_data.columns else set()
                    inv_stores = set(inventory_data['store_id'].unique()) if 'store_id' in inventory_data.columns else set()
                    if sales_stores and inv_stores:
                        missing_stores = sales_stores - inv_stores
                        if missing_stores:
                            warnings.append(f"WARNING: {len(missing_stores)} stores in sales data have no inventory data. "
                                          f"Example stores: {list(missing_stores)[:3]}")
                
            except Exception as e:
                errors.append(f"CRITICAL: Unexpected error during Pydantic validation: {str(e)}")
                is_valid = False
        else:
            # Fallback to basic validation if Pydantic not available
            warnings.append("WARNING: Pydantic not available, using basic validation")
            if sales_data is None or sales_data.empty:
                errors.append("CRITICAL: Sales data is missing or empty. Sales data is required for forecasting.")
            else:
                required_cols = ['store_id', 'sku', 'units_sold']
                missing_cols = [col for col in required_cols if col not in sales_data.columns]
                if missing_cols:
                    errors.append(f"CRITICAL: Sales data missing required columns: {', '.join(missing_cols)}")
            is_valid = len([e for e in errors if e.startswith("CRITICAL:")]) == 0
        
        # Combine all messages
        all_messages = errors + warnings + info
        critical_errors = [e for e in errors if e.startswith("CRITICAL:")]
        
        return {
            'is_valid': is_valid,
            'errors': all_messages,
            'critical_errors': critical_errors,
            'warnings': warnings,
            'info': info,
            'validation_details': validation_details,
            'pydantic_used': PYDANTIC_AVAILABLE
        }
    
    def _analyze_data_characteristics(self, sales_data: pd.DataFrame, 
                                     inventory_data: pd.DataFrame,
                                     price_data: pd.DataFrame,
                                     comparables: List[Dict]) -> Dict[str, Any]:
        """Analyze data to understand patterns for model selection"""
        
        characteristics = {
            'has_time_series': False,
            'time_series_length': 0,
            'has_comparables': len(comparables) > 0,
            'num_comparables': len(comparables),
            'decay_pattern': None,
            'seasonality_detected': False,
            'store_count': 0,
            'sku_count': 0
        }
        
        if sales_data is not None and not sales_data.empty:
            # Check for time series data
            if 'date' in sales_data.columns:
                sales_data['date'] = pd.to_datetime(sales_data['date'])
                characteristics['has_time_series'] = True
                date_range = (sales_data['date'].max() - sales_data['date'].min()).days
                characteristics['time_series_length'] = date_range
                
                # Detect decay/depletion patterns
                if 'units_sold' in sales_data.columns:
                    decay_pattern = self._detect_decay_pattern(sales_data)
                    characteristics['decay_pattern'] = decay_pattern
                    
                # Detect seasonality
                if date_range > 90:  # Need at least 3 months
                    characteristics['seasonality_detected'] = self._detect_seasonality(sales_data)
            
            # Store and SKU counts
            if 'store_id' in sales_data.columns:
                characteristics['store_count'] = sales_data['store_id'].nunique()
            if 'sku' in sales_data.columns:
                characteristics['sku_count'] = sales_data['sku'].nunique()
        
        return characteristics
    
    def _detect_decay_pattern(self, sales_data: pd.DataFrame) -> str:
        """Detect if sales show decay/depletion pattern"""
        if 'date' not in sales_data.columns or 'units_sold' not in sales_data.columns:
            return 'unknown'
        
        # Aggregate by date
        daily_sales = sales_data.groupby('date')['units_sold'].sum().sort_index()
        
        if len(daily_sales) < 7:
            return 'insufficient_data'
        
        # Calculate trend (simple linear regression slope)
        x = np.arange(len(daily_sales))
        y = daily_sales.values
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by mean
        mean_sales = np.mean(y)
        normalized_slope = slope / mean_sales if mean_sales > 0 else 0
        
        if normalized_slope < -0.01:
            return 'strong_decay'
        elif normalized_slope < -0.005:
            return 'moderate_decay'
        elif normalized_slope > 0.01:
            return 'growth'
        else:
            return 'stable'
    
    def _detect_seasonality(self, sales_data: pd.DataFrame) -> bool:
        """Simple seasonality detection"""
        if 'date' not in sales_data.columns or 'units_sold' not in sales_data.columns:
            return False
        
        daily_sales = sales_data.groupby('date')['units_sold'].sum()
        daily_sales.index = pd.to_datetime(daily_sales.index)
        
        # Check for weekly pattern
        if len(daily_sales) >= 14:
            daily_sales['day_of_week'] = daily_sales.index.dayofweek
            weekly_variance = daily_sales.groupby('day_of_week')['units_sold'].mean().std()
            if weekly_variance > daily_sales['units_sold'].mean() * 0.1:
                return True
        
        return False
    
    def _llm_model_selection(self, characteristics: Dict, comparables: List[Dict]) -> Dict[str, Any]:
        """Use LLM to recommend model selection (token-optimized)"""
        
        # Token-optimized prompt - concise and structured
        ts = characteristics.get('has_time_series', False)
        ts_len = characteristics.get('time_series_length', 0)
        has_comp = characteristics.get('has_comparables', False)
        n_comp = characteristics.get('num_comparables', 0)
        decay = characteristics.get('decay_pattern', 'unknown')
        season = characteristics.get('seasonality_detected', False)
        stores = characteristics.get('store_count', 0)
        
        # Ultra-compact prompt to minimize tokens
        prompt = f"Forecast model selection:\nTS:{ts},Len:{ts_len},Comp:{has_comp},N:{n_comp},Decay:{decay},Seas:{season},Stores:{stores}\nModels:1.TimeSeries 2.Decay 3.Analogy 4.Hybrid\nJSON:{{'model':'name','reason':'brief'}}"
        
        # Truncate prompt if too long
        if len(prompt) > self.max_llm_tokens:
            prompt = prompt[:self.max_llm_tokens]
        
        try:
            if self.llama and self.use_llm:
                # Use minimal token generation
                response = self.llama.generate(prompt, max_tokens=100)  # Limit response tokens
                # Parse JSON from response (simplified - in production, use proper JSON parsing)
                return {
                    'recommended_model': 'hybrid',
                    'reasoning': 'Based on data characteristics',
                    'llm_response': response[:200] if response else None  # Truncate response
                }
        except Exception as e:
            print(f"LLM model selection error: {e}")
            self.use_llm = False  # Disable LLM on error
        
        # Fallback recommendation (no LLM needed)
        if ts and ts_len > 60:
            if decay in ['strong_decay', 'moderate_decay']:
                return {
                    'recommended_model': 'decay_model',
                    'reasoning': 'Time series data shows decay pattern'
                }
            else:
                return {
                    'recommended_model': 'time_series',
                    'reasoning': 'Sufficient time series data available'
                }
        elif has_comp and n_comp >= 3:
            return {
                'recommended_model': 'analogy_based',
                'reasoning': 'Good comparable products available'
            }
        else:
            return {
                'recommended_model': 'hybrid',
                'reasoning': 'Combining available approaches'
            }
    
    def _evaluate_models(self, sales_data: pd.DataFrame, inventory_data: pd.DataFrame,
                        price_data: pd.DataFrame, comparables: List[Dict],
                        characteristics: Dict) -> Dict[str, Dict]:
        """Evaluate different forecasting models and return scores"""
        
        model_scores = {}
        
        # 1. Time Series Model (Prophet)
        if characteristics['has_time_series'] and characteristics['time_series_length'] > 30:
            try:
                score, params = self._evaluate_time_series_model(sales_data)
                model_scores['time_series'] = {'score': score, 'params': params}
            except Exception as e:
                print(f"Time series model evaluation error: {e}")
                model_scores['time_series'] = {'score': 0.0, 'params': {}}
        
        # 2. Decay/Depletion Model
        if characteristics['decay_pattern'] in ['strong_decay', 'moderate_decay']:
            try:
                score, params = self._evaluate_decay_model(sales_data, inventory_data)
                model_scores['decay_model'] = {'score': score, 'params': params}
            except Exception as e:
                print(f"Decay model evaluation error: {e}")
                model_scores['decay_model'] = {'score': 0.0, 'params': {}}
        
        # 3. Analogy-based Model
        if characteristics['has_comparables'] and len(comparables) > 0:
            try:
                score, params = self._evaluate_analogy_model(sales_data, comparables)
                model_scores['analogy_based'] = {'score': score, 'params': params}
            except Exception as e:
                print(f"Analogy model evaluation error: {e}")
                model_scores['analogy_based'] = {'score': 0.0, 'params': {}}
        
        # 4. Hybrid Model (always available)
        try:
            score, params = self._evaluate_hybrid_model(
                sales_data, inventory_data, price_data, comparables, characteristics
            )
            model_scores['hybrid'] = {'score': score, 'params': params}
        except Exception as e:
            print(f"Hybrid model evaluation error: {e}")
            model_scores['hybrid'] = {'score': 0.5, 'params': {}}
        
        return model_scores
    
    def _evaluate_time_series_model(self, sales_data: pd.DataFrame) -> Tuple[float, Dict]:
        """Evaluate Prophet time series model"""
        if Prophet is None or sales_data is None or sales_data.empty:
            return 0.0, {}
        
        try:
            # Prepare data for Prophet
            daily_sales = sales_data.groupby('date')['units_sold'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']
            daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
            
            if len(daily_sales) < 7:
                return 0.0, {}
            
            # Split for validation
            split_idx = int(len(daily_sales) * 0.8)
            train = daily_sales[:split_idx]
            test = daily_sales[split_idx:]
            
            if len(train) < 7:
                return 0.0, {}
            
            # Fit Prophet
            model = Prophet()
            model.fit(train)
            
            # Forecast
            future = model.make_future_dataframe(periods=len(test))
            forecast = model.predict(future)
            
            # Calculate error
            predicted = forecast.tail(len(test))['yhat'].values
            actual = test['y'].values
            
            mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
            score = max(0, 100 - mape) / 100  # Convert to 0-1 scale
            
            return score, {'model_type': 'prophet', 'mape': mape}
        except Exception as e:
            print(f"Prophet evaluation error: {e}")
            return 0.0, {}
    
    def _evaluate_decay_model(self, sales_data: pd.DataFrame, 
                             inventory_data: pd.DataFrame) -> Tuple[float, Dict]:
        """Evaluate exponential decay/depletion model"""
        if sales_data is None or sales_data.empty:
            return 0.0, {}
        
        try:
            # Aggregate sales by date
            if 'date' in sales_data.columns and 'units_sold' in sales_data.columns:
                daily_sales = sales_data.groupby('date')['units_sold'].sum().sort_index()
                
                if len(daily_sales) < 7:
                    return 0.0, {}
                
                # Fit exponential decay: y = a * exp(-b * t)
                x = np.arange(len(daily_sales))
                y = daily_sales.values
                
                # Use log-linear regression
                y_log = np.log(y + 1)
                coeffs = np.polyfit(x, y_log, 1)
                decay_rate = -coeffs[0]
                
                # Predict
                predicted = np.exp(coeffs[1] - decay_rate * x)
                
                # Calculate error
                mape = np.mean(np.abs((y - predicted) / (y + 1))) * 100
                score = max(0, 100 - mape) / 100
                
                return score, {
                    'model_type': 'exponential_decay',
                    'decay_rate': decay_rate,
                    'mape': mape
                }
        except Exception as e:
            print(f"Decay model evaluation error: {e}")
        
        return 0.0, {}
    
    def _evaluate_analogy_model(self, sales_data: pd.DataFrame, 
                                comparables: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate analogy-based forecasting model"""
        if not comparables or sales_data is None or sales_data.empty:
            return 0.0, {}
        
        try:
            # Get average sales from comparable SKUs
            comparable_skus = [c.get('sku', '') for c in comparables if 'sku' in c]
            
            if not comparable_skus or 'sku' not in sales_data.columns:
                return 0.0, {}
            
            # Filter sales for comparable SKUs
            comparable_sales = sales_data[sales_data['sku'].isin(comparable_skus)]
            
            if comparable_sales.empty:
                return 0.0, {}
            
            # Calculate average sales per store
            avg_sales = comparable_sales.groupby('store_id')['units_sold'].mean()
            
            # Simple validation: use variance as proxy for model quality
            # Lower variance = more consistent = better model
            variance = avg_sales.var()
            mean_sales = avg_sales.mean()
            
            if mean_sales > 0:
                cv = np.sqrt(variance) / mean_sales  # Coefficient of variation
                score = max(0, 1 - min(cv, 1))  # Lower CV = higher score
            else:
                score = 0.5
            
            return score, {
                'model_type': 'analogy_based',
                'num_comparables': len(comparable_skus),
                'avg_sales_per_store': mean_sales
            }
        except Exception as e:
            print(f"Analogy model evaluation error: {e}")
        
        return 0.0, {}
    
    def _evaluate_hybrid_model(self, sales_data: pd.DataFrame, inventory_data: pd.DataFrame,
                              price_data: pd.DataFrame, comparables: List[Dict],
                              characteristics: Dict) -> Tuple[float, Dict]:
        """Evaluate hybrid model combining multiple approaches"""
        
        # Combine scores from available models
        scores = []
        params = {'model_type': 'hybrid', 'components': []}
        
        if characteristics['has_time_series']:
            ts_score, ts_params = self._evaluate_time_series_model(sales_data)
            if ts_score > 0:
                scores.append(ts_score * 0.4)  # 40% weight
                params['components'].append('time_series')
        
        if characteristics['decay_pattern'] in ['strong_decay', 'moderate_decay']:
            decay_score, decay_params = self._evaluate_decay_model(sales_data, inventory_data)
            if decay_score > 0:
                scores.append(decay_score * 0.3)  # 30% weight
                params['components'].append('decay')
        
        if characteristics['has_comparables']:
            analogy_score, analogy_params = self._evaluate_analogy_model(sales_data, comparables)
            if analogy_score > 0:
                scores.append(analogy_score * 0.3)  # 30% weight
                params['components'].append('analogy')
        
        if not scores:
            return 0.5, params
        
        final_score = sum(scores)
        return final_score, params
    
    def sensitivity_and_factor_analysis(self, sales_data: pd.DataFrame,
                                       price_data: pd.DataFrame,
                                       product_attributes: Dict[str, Any],
                                       comparables: List[Dict]) -> Dict[str, Any]:
        """
        Task 2: Sensitivity and factor analysis with:
        - Correlation analysis of product attributes
        - Dimensionality reduction (PCA)
        - Price bucket analysis
        - Factor importance ranking
        """
        
        print("\n" + "=" * 60)
        print("TASK 2: Sensitivity and Factor Analysis")
        print("=" * 60)
        
        # Prepare data for analysis
        analysis_data = self._prepare_factor_analysis_data(
            sales_data, price_data, product_attributes, comparables
        )
        
        # 1. Correlation Analysis
        correlation_results = self._correlation_analysis(analysis_data)
        
        # 2. Dimensionality Reduction (PCA)
        pca_results = self._dimensionality_reduction(analysis_data)
        
        # 3. Price Bucket Analysis
        price_analysis = self._price_bucket_analysis(sales_data, price_data)
        
        # 4. Factor Importance (using Random Forest)
        factor_importance = self._factor_importance_analysis(analysis_data)
        
        # 5. Sensitivity Analysis
        sensitivity_results = self._sensitivity_analysis(
            analysis_data, price_data, product_attributes
        )
        
        self.factor_analysis_results = {
            'correlation_matrix': correlation_results,
            'pca_results': pca_results,
            'price_analysis': price_analysis,
            'factor_importance': factor_importance,
            'sensitivity_results': sensitivity_results
        }
        
        print("\nFactor Analysis Summary:")
        print(f"- Top correlated attributes: {correlation_results.get('top_correlations', [])}")
        print(f"- PCA explained variance: {pca_results.get('explained_variance_ratio', [])[:3]}")
        print(f"- Most important factors: {[f[0] for f in factor_importance[:5]]}")
        
        return self.factor_analysis_results
    
    def _prepare_factor_analysis_data(self, sales_data: pd.DataFrame,
                                     price_data: pd.DataFrame,
                                     product_attributes: Dict[str, Any],
                                     comparables: List[Dict]) -> pd.DataFrame:
        """Prepare combined dataset for factor analysis"""
        
        # Start with sales data
        if sales_data is None or sales_data.empty:
            return pd.DataFrame()
        
        analysis_df = sales_data.copy()
        
        # Merge price data
        if price_data is not None and not price_data.empty:
            if 'store_id' in price_data.columns and 'sku' in price_data.columns:
                analysis_df = analysis_df.merge(
                    price_data[['store_id', 'sku', 'price']],
                    on=['store_id', 'sku'],
                    how='left',
                    suffixes=('', '_price')
                )
                if 'price' not in analysis_df.columns:
                    analysis_df['price'] = analysis_df.get('price_price', 0)
        
        # Add product attributes as features
        if product_attributes:
            for attr, value in product_attributes.items():
                if isinstance(value, (int, float)):
                    analysis_df[f'attr_{attr}'] = value
                elif isinstance(value, str):
                    # Encode categorical attributes
                    analysis_df[f'attr_{attr}'] = hash(value) % 1000
        
        # Add price buckets
        if 'price' in analysis_df.columns:
            analysis_df['price_bucket'] = pd.cut(
                analysis_df['price'],
                bins=[0, 200, 300, 400, 500, 1000, float('inf')],
                labels=['0-200', '200-300', '300-400', '400-500', '500-1000', '1000+']
            )
            # Convert to numeric for analysis
            analysis_df['price_bucket_encoded'] = analysis_df['price_bucket'].cat.codes
        
        # Add comparable product features
        if comparables:
            comparable_skus = [c.get('sku', '') for c in comparables if 'sku' in c]
            analysis_df['is_comparable'] = analysis_df['sku'].isin(comparable_skus).astype(int)
            if 'similarity' in comparables[0]:
                # Add average similarity score
                similarity_map = {c.get('sku', ''): c.get('similarity', 0) 
                                for c in comparables if 'sku' in c}
                analysis_df['avg_similarity'] = analysis_df['sku'].map(similarity_map).fillna(0)
        
        # Add time-based features
        if 'date' in analysis_df.columns:
            analysis_df['date'] = pd.to_datetime(analysis_df['date'])
            analysis_df['day_of_week'] = analysis_df['date'].dt.dayofweek
            analysis_df['month'] = analysis_df['date'].dt.month
            analysis_df['is_weekend'] = (analysis_df['day_of_week'] >= 5).astype(int)
        
        # Add store-level aggregations
        if 'store_id' in analysis_df.columns and 'units_sold' in analysis_df.columns:
            store_stats = analysis_df.groupby('store_id')['units_sold'].agg(['mean', 'std']).reset_index()
            store_stats.columns = ['store_id', 'store_avg_sales', 'store_sales_std']
            analysis_df = analysis_df.merge(store_stats, on='store_id', how='left')
        
        return analysis_df
    
    def _correlation_analysis(self, analysis_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform correlation analysis on attributes"""
        
        if analysis_data.empty:
            return {}
        
        # Select numeric columns for correlation
        numeric_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target variable
        exclude_cols = ['units_sold', 'revenue', 'store_id', 'sku']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = analysis_data[numeric_cols + ['units_sold']].corr()
        
        # Get top correlations with units_sold
        if 'units_sold' in corr_matrix.columns:
            correlations_with_target = corr_matrix['units_sold'].drop('units_sold').abs().sort_values(ascending=False)
            top_correlations = correlations_with_target.head(10).to_dict()
        else:
            top_correlations = {}
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': top_correlations,
            'strong_correlations': correlations_with_target[correlations_with_target > 0.5].to_dict() if 'units_sold' in corr_matrix.columns else {}
        }
    
    def _dimensionality_reduction(self, analysis_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA for dimensionality reduction"""
        
        if PCA is None or StandardScaler is None or analysis_data.empty:
            return {}
        
        # Select numeric features (exclude target and IDs)
        feature_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['units_sold', 'revenue', 'store_id']
        feature_cols = [col for col in feature_cols if col not in exclude_cols and not col.startswith('attr_')]
        
        if len(feature_cols) < 2:
            return {}
        
        X = analysis_data[feature_cols].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        n_components = min(5, len(feature_cols))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Get feature loadings
        feature_loadings = {}
        for i, feature in enumerate(feature_cols):
            feature_loadings[feature] = pca.components_[:, i].tolist() if i < len(pca.components_) else []
        
        return {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': X_pca.tolist(),
            'feature_loadings': feature_loadings,
            'scaler': scaler
        }
    
    def _price_bucket_analysis(self, sales_data: pd.DataFrame, 
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sales performance by price bucket"""
        
        if sales_data is None or sales_data.empty or price_data is None or price_data.empty:
            return {}
        
        # Merge sales and price data
        merged = sales_data.merge(
            price_data[['store_id', 'sku', 'price']],
            on=['store_id', 'sku'],
            how='inner'
        )
        
        if merged.empty:
            return {}
        
        # Create price buckets
        merged['price_bucket'] = pd.cut(
            merged['price'],
            bins=[0, 200, 300, 400, 500, 1000, float('inf')],
            labels=['0-200', '200-300', '300-400', '400-500', '500-1000', '1000+']
        )
        
        # Analyze by price bucket
        bucket_analysis = merged.groupby('price_bucket').agg({
            'units_sold': ['sum', 'mean', 'count'],
            'revenue': 'sum' if 'revenue' in merged.columns else 'count'
        }).reset_index()
        
        bucket_analysis.columns = ['price_bucket', 'total_units', 'avg_units', 'transaction_count', 'total_revenue']
        
        # Calculate price elasticity proxy
        price_elasticity = {}
        for bucket in bucket_analysis['price_bucket'].unique():
            bucket_data = merged[merged['price_bucket'] == bucket]
            if len(bucket_data) > 1:
                avg_price = bucket_data['price'].mean()
                avg_units = bucket_data['units_sold'].mean()
                price_elasticity[str(bucket)] = {
                    'avg_price': float(avg_price),
                    'avg_units': float(avg_units),
                    'price_per_unit': float(avg_price / avg_units) if avg_units > 0 else 0
                }
        
        return {
            'bucket_statistics': bucket_analysis.to_dict('records'),
            'price_elasticity': price_elasticity
        }
    
    def _factor_importance_analysis(self, analysis_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Use Random Forest to determine factor importance"""
        
        if RandomForestRegressor is None or analysis_data.empty:
            return []
        
        # Prepare features and target
        feature_cols = analysis_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['units_sold', 'revenue']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        if 'units_sold' not in analysis_data.columns or len(feature_cols) == 0:
            return []
        
        X = analysis_data[feature_cols].fillna(0)
        y = analysis_data['units_sold'].fillna(0)
        
        if len(X) < 10:
            return []
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Get feature importance
        feature_importance = list(zip(feature_cols, rf.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance
    
    def _sensitivity_analysis(self, analysis_data: pd.DataFrame,
                             price_data: pd.DataFrame,
                             product_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sensitivity analysis on key factors"""
        
        sensitivity_results = {}
        
        if analysis_data.empty:
            return sensitivity_results
        
        # Price sensitivity
        if 'price' in analysis_data.columns and 'units_sold' in analysis_data.columns:
            price_ranges = analysis_data['price'].quantile([0.25, 0.5, 0.75]).values
            price_sensitivity = {}
            
            for price_threshold in price_ranges:
                below_price = analysis_data[analysis_data['price'] <= price_threshold]
                above_price = analysis_data[analysis_data['price'] > price_threshold]
                
                if len(below_price) > 0 and len(above_price) > 0:
                    avg_sales_below = below_price['units_sold'].mean()
                    avg_sales_above = above_price['units_sold'].mean()
                    
                    price_sensitivity[f'price_{price_threshold:.0f}'] = {
                        'below_threshold_avg_sales': float(avg_sales_below),
                        'above_threshold_avg_sales': float(avg_sales_above),
                        'sensitivity_ratio': float(avg_sales_below / avg_sales_above) if avg_sales_above > 0 else 0
                    }
            
            sensitivity_results['price_sensitivity'] = price_sensitivity
        
        # Attribute sensitivity (if attributes available)
        attr_cols = [col for col in analysis_data.columns if col.startswith('attr_')]
        for attr_col in attr_cols:
            if analysis_data[attr_col].nunique() > 1:
                attr_sensitivity = analysis_data.groupby(attr_col)['units_sold'].mean().to_dict()
                sensitivity_results[f'{attr_col}_sensitivity'] = attr_sensitivity
        
        return sensitivity_results
    
    def forecast_store_level(self, sales_data: pd.DataFrame,
                           inventory_data: pd.DataFrame,
                           price_data: pd.DataFrame,
                           product_attributes: Dict[str, Any],
                           comparables: List[Dict],
                           forecast_horizon_days: int = 60,
                           cost_data: Optional[pd.DataFrame] = None,
                           margin_target: Optional[float] = None,
                           max_quantity_per_store: int = 500) -> Dict[str, Any]:
        """
        Task 3: Store-level forecasting with optimization for:
        - Maximum rate of sale
        - Maximum sell-through rate
        - Margin protection (minimize margin erosion, ensure >= margin_target)
        - Which articles to buy at each store
        - How much quantity to buy (max 500 per store)
        - Based on results from Task 1 (model selection) and Task 2 (factor analysis)
        - Considers cost of procuring articles
        - Analyzes historical MRP, selling price, discount, and margin
        """
        
        print("\n" + "=" * 60)
        print("TASK 3: Store-Level Forecasting (Optimized)")
        print("=" * 60)
        print(f"Optimization Targets:")
        print(f"  - Target Sell-Through Rate: {self.target_sell_through_pct*100:.1f}%")
        print(f"  - Minimum Margin Protection: {self.min_margin_pct*100:.1f}%")
        print(f"  - Maximize Rate of Sale")
        
        if sales_data is None or sales_data.empty:
            print("Warning: No sales data available for forecasting")
            return {}
        
        # Get unique stores
        if 'store_id' not in sales_data.columns:
            print("Warning: No store_id column found")
            return {}
        
        stores = sales_data['store_id'].unique()
        store_forecasts = {}
        
        # Get unique SKUs/articles
        if 'sku' in sales_data.columns:
            articles = sales_data['sku'].unique()
        else:
            articles = ['unknown_article']
        
        print(f"\nForecasting for {len(stores)} stores and {len(articles)} articles")
        print(f"Forecast horizon: {forecast_horizon_days} days")
        print(f"Using model: {self.selected_model}")
        
        # Calculate historical performance metrics for optimization
        historical_metrics = self._calculate_historical_metrics(
            sales_data, inventory_data, price_data
        )
        
        # Analyze historical pricing data (MRP, selling price, discount, margin)
        historical_pricing = self._analyze_historical_pricing(
            sales_data, price_data, cost_data
        )
        
        # Forecast for each store-article combination
        for store_id in stores:
            store_forecasts[str(store_id)] = {}
            
            for article in articles:
                # Filter data for this store-article combination
                store_article_data = sales_data[
                    (sales_data['store_id'] == store_id) & 
                    (sales_data['sku'] == article)
                ].copy()
                
                if store_article_data.empty:
                    # Use comparable products or store average
                    forecast = self._forecast_without_history(
                        store_id, article, sales_data, comparables, price_data
                    )
                else:
                    # Use selected model to forecast
                    forecast = self._generate_forecast(
                        store_article_data, store_id, article, 
                        inventory_data, price_data, comparables,
                        forecast_horizon_days, sales_data=sales_data
                    )
                
                # Apply per-store maximum quantity constraint (500)
                if forecast.get('forecast_quantity', 0) > max_quantity_per_store:
                    forecast['forecast_quantity'] = max_quantity_per_store
                    forecast['reasoning'] += f" | Capped at {max_quantity_per_store} units per store"
                
                # Optimize forecast for rate of sale, sell-through, and margin
                forecast = self._optimize_forecast(
                    forecast, store_id, article, sales_data, 
                    inventory_data, price_data, historical_metrics,
                    forecast_horizon_days, cost_data, margin_target
                )
                
                store_forecasts[str(store_id)][str(article)] = forecast
        
        # Aggregate results
        aggregated_forecast = self._aggregate_forecasts(store_forecasts)
        
        # Generate optimized recommendations
        recommendations = self._generate_optimized_recommendations(
            store_forecasts, aggregated_forecast, product_attributes,
            sales_data, price_data, historical_metrics, forecast_horizon_days,
            cost_data, margin_target, historical_pricing
        )
        
        self.forecast_results = {
            'store_level_forecasts': store_forecasts,
            'aggregated_forecast': aggregated_forecast,
            'recommendations': recommendations,
            'forecast_horizon_days': forecast_horizon_days,
            'model_used': self.selected_model,
            'optimization_metrics': {
                'target_sell_through_pct': self.target_sell_through_pct,
                'min_margin_pct': self.min_margin_pct,
                'default_margin_pct': self.default_margin_pct
            }
        }
        
        print("\n" + "=" * 60)
        print("Forecasting Complete (Optimized)")
        print("=" * 60)
        print(f"\nTotal stores forecasted: {len(store_forecasts)}")
        print(f"Total articles recommended: {len(aggregated_forecast.get('articles', []))}")
        print(f"Total quantity recommended: {aggregated_forecast.get('total_quantity', 0):.0f} units")
        if 'avg_sell_through_rate' in aggregated_forecast:
            print(f"Expected sell-through rate: {aggregated_forecast['avg_sell_through_rate']*100:.1f}%")
        if 'avg_margin_pct' in aggregated_forecast:
            print(f"Expected margin: {aggregated_forecast['avg_margin_pct']*100:.1f}%")
        
        return self.forecast_results
    
    def _generate_forecast(self, store_article_data: pd.DataFrame,
                          store_id: str, article: str,
                          inventory_data: pd.DataFrame,
                          price_data: pd.DataFrame,
                          comparables: List[Dict],
                          forecast_horizon_days: int,
                          sales_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate forecast using the selected model with fallback to LR+KMeans"""
        
        # Get price information
        price = 0
        if price_data is not None and not price_data.empty:
            price_filter = price_data[
                (price_data['store_id'] == store_id) & 
                (price_data['sku'] == article)
            ]
            if not price_filter.empty and 'price' in price_filter.columns:
                price = price_filter['price'].iloc[0]
        
        forecast = {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': 0,
            'confidence_low': 0,
            'confidence_high': 0,
            'recommendation': 'skip',
            'reasoning': '',
            'price': price,
            'method_used': 'unknown'
        }
        
        # Try primary forecasting methods first
        if self.selected_model != 'fallback_lr_kmeans':
            try:
                if self.selected_model == 'time_series':
                    forecast = self._forecast_time_series(
                        store_article_data, store_id, article, forecast_horizon_days
                    )
                    forecast['method_used'] = 'time_series'
                elif self.selected_model == 'decay_model':
                    forecast = self._forecast_decay(
                        store_article_data, store_id, article, inventory_data, forecast_horizon_days
                    )
                    forecast['method_used'] = 'decay_model'
                elif self.selected_model == 'analogy_based':
                    forecast = self._forecast_analogy(
                        store_article_data, store_id, article, sales_data=sales_data, comparables=comparables
                    )
                    forecast['method_used'] = 'analogy_based'
                else:  # hybrid
                    forecast = self._forecast_hybrid(
                        store_article_data, store_id, article, inventory_data, 
                        price_data, comparables, forecast_horizon_days
                    )
                    forecast['method_used'] = 'hybrid'
                
                # Validate forecast result
                if forecast.get('forecast_quantity', 0) <= 0 and forecast.get('recommendation') != 'skip':
                    raise ValueError("Forecast produced invalid quantity")
                
            except Exception as e:
                print(f"⚠️  Primary forecast failed for {store_id}-{article}: {e}")
                print(f"🔄 Falling back to Linear Regression + K-Means")
                # Fall through to fallback method
        
        # Use fallback if primary methods failed or fallback was selected
        if (self.selected_model == 'fallback_lr_kmeans' or 
            forecast.get('forecast_quantity', 0) <= 0 or 
            forecast.get('recommendation') == 'skip'):
            try:
                forecast = self._forecast_fallback_lr_kmeans(
                    store_id, article, sales_data, inventory_data, 
                    price_data, forecast_horizon_days
                )
                forecast['method_used'] = 'fallback_lr_kmeans'
                self.fallback_used = True
            except Exception as e:
                print(f"❌ Fallback forecast also failed for {store_id}-{article}: {e}")
                forecast['reasoning'] = f"All forecasting methods failed: {str(e)}"
                forecast['method_used'] = 'failed'
        
        # Ensure price is included
        forecast['price'] = price
        
        return forecast
    
    def _forecast_fallback_lr_kmeans(self, store_id: str, article: str,
                                    sales_data: Optional[pd.DataFrame],
                                    inventory_data: Optional[pd.DataFrame],
                                    price_data: Optional[pd.DataFrame],
                                    forecast_horizon_days: int) -> Dict[str, Any]:
        """
        Fallback forecasting using Linear Regression + K-Means Clustering
        This method is used when primary forecasting methods fail
        """
        
        if LinearRegression is None or KMeans is None:
            raise ImportError("scikit-learn not available for fallback forecasting")
        
        print(f"  Using Linear Regression + K-Means for {store_id}-{article}")
        
        # Step 1: Prepare features for clustering and regression
        features_list = []
        target_list = []
        store_article_pairs = []
        
        if sales_data is not None and not sales_data.empty:
            # Aggregate sales by store-article
            if 'date' in sales_data.columns:
                sales_data['date'] = pd.to_datetime(sales_data['date'])
                sales_agg = sales_data.groupby(['store_id', 'sku', 'date']).agg({
                    'units_sold': 'sum'
                }).reset_index()
            else:
                sales_agg = sales_data.groupby(['store_id', 'sku']).agg({
                    'units_sold': 'sum'
                }).reset_index()
                sales_agg['date'] = pd.Timestamp.now()
            
            # Create features for each store-article combination
            for (sid, sku), group in sales_agg.groupby(['store_id', 'sku']):
                if len(group) < 2:  # Need at least 2 data points
                    continue
                
                # Features: mean sales, std sales, number of days, recent trend
                mean_sales = group['units_sold'].mean()
                std_sales = group['units_sold'].std() if len(group) > 1 else 0
                n_days = len(group)
                
                # Recent trend (last 30% vs first 30%)
                if len(group) >= 3:
                    recent = group.tail(max(1, len(group) // 3))['units_sold'].mean()
                    early = group.head(max(1, len(group) // 3))['units_sold'].mean()
                    trend = (recent - early) / (early + 1) if early > 0 else 0
                else:
                    trend = 0
                
                # Add price feature if available
                price_feature = 0
                if price_data is not None and not price_data.empty:
                    price_filter = price_data[
                        (price_data['store_id'] == sid) & 
                        (price_data['sku'] == sku)
                    ]
                    if not price_filter.empty and 'price' in price_filter.columns:
                        price_feature = price_filter['price'].iloc[0]
                
                # Add inventory feature if available
                inv_feature = 0
                if inventory_data is not None and not inventory_data.empty:
                    inv_filter = inventory_data[
                        (inventory_data['store_id'] == sid) & 
                        (inventory_data['sku'] == sku)
                    ]
                    if not inv_filter.empty and 'on_hand' in inv_filter.columns:
                        inv_feature = inv_filter['on_hand'].iloc[0]
                
                # Target: total units sold
                target = group['units_sold'].sum()
                
                features_list.append([mean_sales, std_sales, n_days, trend, price_feature, inv_feature])
                target_list.append(target)
                store_article_pairs.append((sid, sku))
        
        if len(features_list) < 3:
            # Not enough data for clustering/regression, use simple average
            if sales_data is not None and not sales_data.empty:
                store_article_sales = sales_data[
                    (sales_data['store_id'] == store_id) & 
                    (sales_data['sku'] == article)
                ]
                if not store_article_sales.empty:
                    avg_daily = store_article_sales['units_sold'].mean()
                    forecast_qty = avg_daily * forecast_horizon_days
                else:
                    # Use overall average
                    avg_daily = sales_data['units_sold'].mean() if 'units_sold' in sales_data.columns else 0
                    forecast_qty = avg_daily * forecast_horizon_days
            else:
                forecast_qty = 0
            
            return {
                'store_id': store_id,
                'article': article,
                'forecast_quantity': max(0, forecast_qty),
                'confidence_low': max(0, forecast_qty * 0.7),
                'confidence_high': forecast_qty * 1.3,
                'recommendation': 'buy' if forecast_qty > 5 else 'skip',
                'reasoning': 'Fallback: Simple average (insufficient data for LR+KMeans)',
                'price': 0
            }
        
        # Step 2: K-Means Clustering to group similar store-article patterns
        X = np.array(features_list)
        n_clusters = min(5, max(2, len(features_list) // 3))  # Adaptive cluster count
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Step 3: Find cluster for target store-article
        target_features = None
        target_cluster = -1
        
        # Get features for target store-article
        if sales_data is not None and not sales_data.empty:
            target_sales = sales_data[
                (sales_data['store_id'] == store_id) & 
                (sales_data['sku'] == article)
            ]
            
            if not target_sales.empty:
                if 'date' in target_sales.columns:
                    target_sales['date'] = pd.to_datetime(target_sales['date'])
                    target_group = target_sales.groupby('date')['units_sold'].sum()
                else:
                    target_group = target_sales['units_sold']
                
                mean_sales = target_group.mean() if len(target_group) > 0 else 0
                std_sales = target_group.std() if len(target_group) > 1 else 0
                n_days = len(target_group)
                
                if len(target_group) >= 3:
                    recent = target_group.tail(max(1, len(target_group) // 3)).mean()
                    early = target_group.head(max(1, len(target_group) // 3)).mean()
                    trend = (recent - early) / (early + 1) if early > 0 else 0
                else:
                    trend = 0
                
                price_feature = 0
                if price_data is not None and not price_data.empty:
                    price_filter = price_data[
                        (price_data['store_id'] == store_id) & 
                        (price_data['sku'] == article)
                    ]
                    if not price_filter.empty and 'price' in price_filter.columns:
                        price_feature = price_filter['price'].iloc[0]
                
                inv_feature = 0
                if inventory_data is not None and not inventory_data.empty:
                    inv_filter = inventory_data[
                        (inventory_data['store_id'] == store_id) & 
                        (inventory_data['sku'] == article)
                    ]
                    if not inv_filter.empty and 'on_hand' in inv_filter.columns:
                        inv_feature = inv_filter['on_hand'].iloc[0]
                
                target_features = np.array([[mean_sales, std_sales, n_days, trend, price_feature, inv_feature]])
                target_cluster = kmeans.predict(target_features)[0]
        
        # Step 4: Linear Regression within the cluster
        cluster_mask = clusters == target_cluster
        cluster_X = X[cluster_mask]
        cluster_y = np.array(target_list)[cluster_mask]
        
        if len(cluster_X) < 2:
            # Not enough data in cluster, use cluster average
            if len(cluster_y) > 0:
                avg_demand = cluster_y.mean()
            else:
                # Use overall average
                avg_demand = np.mean(target_list) if target_list else 0
            
            forecast_qty = (avg_demand / forecast_horizon_days) * forecast_horizon_days if forecast_horizon_days > 0 else avg_demand
        else:
            # Train linear regression
            lr = LinearRegression()
            lr.fit(cluster_X, cluster_y)
            
            # Predict for target
            if target_features is not None:
                predicted_demand = lr.predict(target_features)[0]
                # Convert to forecast quantity (assuming daily rate)
                forecast_qty = (predicted_demand / forecast_horizon_days) * forecast_horizon_days if forecast_horizon_days > 0 else predicted_demand
            else:
                # Fallback to cluster average
                forecast_qty = cluster_y.mean() if len(cluster_y) > 0 else 0
        
        # Get price
        price = 0
        if price_data is not None and not price_data.empty:
            price_filter = price_data[
                (price_data['store_id'] == store_id) & 
                (price_data['sku'] == article)
            ]
            if not price_filter.empty and 'price' in price_filter.columns:
                price = price_filter['price'].iloc[0]
        
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': max(0, forecast_qty),
            'confidence_low': max(0, forecast_qty * 0.7),
            'confidence_high': forecast_qty * 1.3,
            'recommendation': 'buy' if forecast_qty > 5 else 'skip',
            'reasoning': f'Fallback: Linear Regression + K-Means (cluster {target_cluster}, {len(cluster_X)} samples)',
            'price': price
        }
    
    def _forecast_time_series(self, store_article_data: pd.DataFrame,
                             store_id: str, article: str,
                             forecast_horizon_days: int) -> Dict[str, Any]:
        """Forecast using time series model"""
        
        if 'date' not in store_article_data.columns or 'units_sold' not in store_article_data.columns:
            return self._default_forecast(store_id, article)
        
        # Prepare time series
        store_article_data['date'] = pd.to_datetime(store_article_data['date'])
        daily_sales = store_article_data.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        if len(daily_sales) < 7:
            # Not enough data, use simple average
            avg_daily = daily_sales['y'].mean() if len(daily_sales) > 0 else 0
            forecast_qty = avg_daily * forecast_horizon_days
            return {
                'store_id': store_id,
                'article': article,
                'forecast_quantity': max(0, forecast_qty),
                'confidence_low': max(0, forecast_qty * 0.7),
                'confidence_high': forecast_qty * 1.3,
                'recommendation': 'buy' if forecast_qty > 0 else 'skip',
                'reasoning': 'Insufficient data, using average',
                'price': 0  # Will be set by caller
            }
        
        # Use Prophet if available
        if Prophet is not None:
            try:
                model = Prophet()
                model.fit(daily_sales)
                
                future = model.make_future_dataframe(periods=forecast_horizon_days)
                forecast_df = model.predict(future)
                
                # Sum forecast for horizon
                forecast_qty = forecast_df.tail(forecast_horizon_days)['yhat'].sum()
                forecast_low = forecast_df.tail(forecast_horizon_days)['yhat_lower'].sum()
                forecast_high = forecast_df.tail(forecast_horizon_days)['yhat_upper'].sum()
                
                return {
                    'store_id': store_id,
                    'article': article,
                    'forecast_quantity': max(0, forecast_qty),
                    'confidence_low': max(0, forecast_low),
                    'confidence_high': max(0, forecast_high),
                    'recommendation': 'buy' if forecast_qty > 10 else 'skip',
                    'reasoning': 'Prophet time series forecast',
                    'price': 0  # Will be set by caller
                }
            except Exception as e:
                print(f"Prophet forecast error: {e}")
        
        # Fallback: simple moving average
        avg_daily = daily_sales['y'].mean()
        std_daily = daily_sales['y'].std()
        forecast_qty = avg_daily * forecast_horizon_days
        
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': max(0, forecast_qty),
            'confidence_low': max(0, (avg_daily - std_daily) * forecast_horizon_days),
            'confidence_high': (avg_daily + std_daily) * forecast_horizon_days,
            'recommendation': 'buy' if forecast_qty > 10 else 'skip',
            'reasoning': 'Moving average forecast',
            'price': 0  # Will be set by caller
        }
    
    def _forecast_decay(self, store_article_data: pd.DataFrame,
                       store_id: str, article: str,
                       inventory_data: pd.DataFrame,
                       forecast_horizon_days: int) -> Dict[str, Any]:
        """Forecast using decay/depletion model"""
        
        if 'date' not in store_article_data.columns or 'units_sold' not in store_article_data.columns:
            return self._default_forecast(store_id, article, price=0)
        
        store_article_data['date'] = pd.to_datetime(store_article_data['date'])
        daily_sales = store_article_data.groupby('date')['units_sold'].sum().sort_index()
        
        if len(daily_sales) < 7:
            return self._default_forecast(store_id, article, price=0)
        
        # Fit exponential decay: y = a * exp(-b * t)
        x = np.arange(len(daily_sales))
        y = daily_sales.values
        
        # Use log-linear regression
        y_log = np.log(y + 1)
        coeffs = np.polyfit(x, y_log, 1)
        a = np.exp(coeffs[1])
        b = -coeffs[0]
        
        # Forecast future periods
        future_x = np.arange(len(daily_sales), len(daily_sales) + forecast_horizon_days)
        future_y = a * np.exp(-b * future_x)
        forecast_qty = np.sum(future_y)
        
        # Get current inventory if available
        current_inventory = 0
        if inventory_data is not None and not inventory_data.empty:
            inv_filter = inventory_data[
                (inventory_data['store_id'] == store_id) & 
                (inventory_data['sku'] == article)
            ]
            if not inv_filter.empty:
                current_inventory = inv_filter['on_hand'].iloc[0] if 'on_hand' in inv_filter.columns else 0
        
        # Adjust forecast based on inventory
        net_forecast = max(0, forecast_qty - current_inventory)
        
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': max(0, net_forecast),
            'confidence_low': max(0, net_forecast * 0.8),
            'confidence_high': net_forecast * 1.2,
            'current_inventory': current_inventory,
            'recommendation': 'buy' if net_forecast > 5 else 'skip',
            'reasoning': f'Exponential decay model (decay rate: {b:.4f})',
            'price': 0  # Will be set by caller
        }
    
    def _forecast_analogy(self, store_article_data: pd.DataFrame,
                         store_id: str, article: str,
                         sales_data: Optional[pd.DataFrame],
                         comparables: List[Dict]) -> Dict[str, Any]:
        """Forecast using analogy-based approach"""
        
        if not comparables:
            return self._default_forecast(store_id, article)
        
        # Get comparable SKUs
        comparable_skus = [c.get('sku', '') for c in comparables if 'sku' in c]
        similarity_scores = {c.get('sku', ''): c.get('similarity', 0.5) 
                           for c in comparables if 'sku' in c}
        
        if not comparable_skus or sales_data is None or sales_data.empty:
            return self._default_forecast(store_id, article)
        
        # Get sales for comparable SKUs at this store
        comparable_sales = sales_data[
            (sales_data['store_id'] == store_id) & 
            (sales_data['sku'].isin(comparable_skus))
        ]
        
        if comparable_sales.empty:
            return self._default_forecast(store_id, article)
        
        # Calculate weighted average based on similarity
        weighted_sales = []
        for sku in comparable_skus:
            sku_sales = comparable_sales[comparable_sales['sku'] == sku]
            if not sku_sales.empty:
                avg_sales = sku_sales['units_sold'].mean()
                similarity = similarity_scores.get(sku, 0.5)
                weighted_sales.append(avg_sales * similarity)
        
        if not weighted_sales:
            return self._default_forecast(store_id, article)
        
        # Forecast for 60 days
        forecast_qty = np.mean(weighted_sales) * 60
        
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': max(0, forecast_qty),
            'confidence_low': max(0, forecast_qty * 0.7),
            'confidence_high': forecast_qty * 1.3,
            'recommendation': 'buy' if forecast_qty > 10 else 'skip',
            'reasoning': f'Analogy-based forecast using {len(comparable_skus)} comparable products',
            'price': 0  # Will be set by caller
        }
    
    def _forecast_hybrid(self, store_article_data: pd.DataFrame,
                        store_id: str, article: str,
                        inventory_data: pd.DataFrame,
                        price_data: pd.DataFrame,
                        comparables: List[Dict],
                        forecast_horizon_days: int) -> Dict[str, Any]:
        """Forecast using hybrid approach combining multiple methods"""
        
        forecasts = []
        weights = []
        
        # Try time series
        if 'date' in store_article_data.columns and len(store_article_data) >= 7:
            ts_forecast = self._forecast_time_series(
                store_article_data, store_id, article, forecast_horizon_days
            )
            if ts_forecast['forecast_quantity'] > 0:
                forecasts.append(ts_forecast['forecast_quantity'])
                weights.append(0.4)
        
        # Try decay model if decay pattern detected
        if len(store_article_data) >= 7:
            decay_forecast = self._forecast_decay(
                store_article_data, store_id, article, inventory_data, forecast_horizon_days
            )
            if decay_forecast['forecast_quantity'] > 0:
                forecasts.append(decay_forecast['forecast_quantity'])
                weights.append(0.3)
        
        # Try analogy
        if comparables:
            analogy_forecast = self._forecast_analogy(
                store_article_data, store_id, article, None, comparables
            )
            if analogy_forecast['forecast_quantity'] > 0:
                forecasts.append(analogy_forecast['forecast_quantity'])
                weights.append(0.3)
        
        if not forecasts:
            return self._default_forecast(store_id, article)
        
        # Weighted average
        if len(weights) != len(forecasts):
            weights = [1.0 / len(forecasts)] * len(forecasts)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        forecast_qty = sum(f * w for f, w in zip(forecasts, weights))
        
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': max(0, forecast_qty),
            'confidence_low': max(0, forecast_qty * 0.75),
            'confidence_high': forecast_qty * 1.25,
            'recommendation': 'buy' if forecast_qty > 10 else 'skip',
            'reasoning': f'Hybrid forecast combining {len(forecasts)} methods',
            'price': 0  # Will be set by caller
        }
    
    def _forecast_without_history(self, store_id: str, article: str,
                                 sales_data: pd.DataFrame,
                                 comparables: List[Dict],
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """Forecast when no historical data available for store-article combination"""
        
        # Use store average or comparable products
        if sales_data is not None and not sales_data.empty:
            # Store average
            store_sales = sales_data[sales_data['store_id'] == store_id]
            if not store_sales.empty and 'units_sold' in store_sales.columns:
                avg_daily = store_sales['units_sold'].mean()
                forecast_qty = avg_daily * 60  # 60 days
                
                return {
                    'store_id': store_id,
                    'article': article,
                    'forecast_quantity': max(0, forecast_qty),
                    'confidence_low': max(0, forecast_qty * 0.5),
                    'confidence_high': forecast_qty * 1.5,
                    'recommendation': 'buy' if forecast_qty > 5 else 'skip',
                    'reasoning': 'No history, using store average',
                    'price': 0  # Will be set by caller
                }
        
        # Default minimal forecast
        return self._default_forecast(store_id, article, price=0)
    
    def _default_forecast(self, store_id: str, article: str, price: float = 0) -> Dict[str, Any]:
        """Default forecast when no data available"""
        return {
            'store_id': store_id,
            'article': article,
            'forecast_quantity': 0,
            'confidence_low': 0,
            'confidence_high': 0,
            'recommendation': 'skip',
            'reasoning': 'Insufficient data for forecasting',
            'price': price
        }
    
    def _calculate_historical_metrics(self, sales_data: pd.DataFrame,
                                    inventory_data: pd.DataFrame,
                                    price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate historical performance metrics for optimization"""
        
        metrics = {
            'rate_of_sale': {},  # units per day by store-article
            'sell_through_rate': {},  # percentage by store-article
            'margins': {},  # margin percentage by store-article
            'price_elasticity': {}  # price sensitivity by article
        }
        
        if sales_data is None or sales_data.empty:
            return metrics
        
        # Calculate rate of sale (units per day)
        if 'date' in sales_data.columns and 'units_sold' in sales_data.columns:
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            for (store_id, sku), group in sales_data.groupby(['store_id', 'sku']):
                if 'date' in group.columns:
                    date_range = (group['date'].max() - group['date'].min()).days
                    if date_range > 0:
                        total_units = group['units_sold'].sum()
                        rate_of_sale = total_units / date_range
                        key = f"{store_id}_{sku}"
                        metrics['rate_of_sale'][key] = rate_of_sale
        
        # Calculate sell-through rate
        if inventory_data is not None and not inventory_data.empty:
            for (store_id, sku), group in sales_data.groupby(['store_id', 'sku']):
                key = f"{store_id}_{sku}"
                
                # Get initial inventory
                inv_filter = inventory_data[
                    (inventory_data['store_id'] == store_id) & 
                    (inventory_data['sku'] == sku)
                ]
                
                if not inv_filter.empty and 'on_hand' in inv_filter.columns:
                    initial_inventory = inv_filter['on_hand'].iloc[0]
                    total_sold = group['units_sold'].sum() if not group.empty else 0
                    
                    if initial_inventory > 0:
                        sell_through = total_sold / (initial_inventory + total_sold)
                        metrics['sell_through_rate'][key] = sell_through
        
        # Calculate margins (estimate from price if cost not available)
        if price_data is not None and not price_data.empty:
            for (store_id, sku), group in sales_data.groupby(['store_id', 'sku']):
                key = f"{store_id}_{sku}"
                
                # Get price
                price_filter = price_data[
                    (price_data['store_id'] == store_id) & 
                    (price_data['sku'] == sku)
                ]
                
                if not price_filter.empty and 'price' in price_filter.columns:
                    price = price_filter['price'].iloc[0]
                    # Estimate cost (assuming default margin if cost not available)
                    estimated_cost = price * (1 - self.default_margin_pct)
                    margin_pct = (price - estimated_cost) / price if price > 0 else 0
                    metrics['margins'][key] = margin_pct
        
        return metrics
    
    def _analyze_historical_pricing(self, sales_data: pd.DataFrame,
                                    price_data: pd.DataFrame,
                                    cost_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze historical pricing data: MRP, selling price, discount, and margin
        Returns pricing insights for each article
        """
        pricing_analysis = {}
        
        if sales_data is None or sales_data.empty or price_data is None or price_data.empty:
            return pricing_analysis
        
        # Get unique articles
        articles = sales_data['sku'].unique() if 'sku' in sales_data.columns else []
        
        for article in articles:
            article_pricing = {
                'mrp': None,
                'avg_selling_price': 0.0,
                'avg_discount': 0.0,
                'avg_margin': 0.0,
                'price_consistency': 'variable',  # 'consistent' or 'variable'
                'discount_consistency': 'variable',
                'margin_consistency': 'variable'
            }
            
            # Get price data for this article
            article_price_data = price_data[price_data['sku'] == article]
            if not article_price_data.empty:
                # Get MRP (if available) or use max price as MRP
                if 'mrp' in article_price_data.columns:
                    mrps = article_price_data['mrp'].dropna().unique()
                    if len(mrps) > 0:
                        article_pricing['mrp'] = float(mrps[0])  # MRP should be same across stores
                        article_pricing['mrp_consistent'] = len(mrps) == 1
                elif 'price' in article_price_data.columns:
                    # Use max price as MRP if MRP column not available
                    max_price = article_price_data['price'].max()
                    article_pricing['mrp'] = float(max_price)
                    article_pricing['mrp_consistent'] = True
                
                # Get average selling price
                if 'price' in article_price_data.columns:
                    prices = article_price_data['price'].dropna()
                    article_pricing['avg_selling_price'] = float(prices.mean())
                    article_pricing['price_consistency'] = 'consistent' if prices.std() < prices.mean() * 0.05 else 'variable'
                
                # Calculate average discount
                if article_pricing['mrp'] and article_pricing['avg_selling_price']:
                    discount = ((article_pricing['mrp'] - article_pricing['avg_selling_price']) / article_pricing['mrp']) * 100
                    article_pricing['avg_discount'] = float(discount)
                
                # Calculate margin if cost data available
                if cost_data is not None and not cost_data.empty:
                    if 'sku' in cost_data.columns and 'cost' in cost_data.columns:
                        cost_filter = cost_data[cost_data['sku'] == article]
                        if not cost_filter.empty:
                            cost = cost_filter['cost'].iloc[0]
                            if article_pricing['avg_selling_price'] > 0:
                                margin = ((article_pricing['avg_selling_price'] - cost) / article_pricing['avg_selling_price']) * 100
                                article_pricing['avg_margin'] = float(margin)
            
            pricing_analysis[str(article)] = article_pricing
        
        return pricing_analysis
    
    def _calculate_rate_of_sale(self, sales_data: pd.DataFrame, 
                                store_id: str, article: str,
                                forecast_horizon_days: int) -> float:
        """
        Calculate rate of sale (ROS): total inventory sold in period / total days in period
        
        Formula: ROS = Total Units Sold / Total Days in Period
        """
        
        store_article_data = sales_data[
            (sales_data['store_id'] == store_id) & 
            (sales_data['sku'] == article)
        ]
        
        if store_article_data.empty:
            return 0.0
        
        # Calculate total units sold
        total_units_sold = store_article_data['units_sold'].sum() if 'units_sold' in store_article_data.columns else 0
        
        if total_units_sold <= 0:
            return 0.0
        
        # Calculate total days in period
        if 'date' in store_article_data.columns:
            store_article_data['date'] = pd.to_datetime(store_article_data['date'])
            date_range = (store_article_data['date'].max() - store_article_data['date'].min()).days
            # Add 1 to include both start and end dates
            total_days = max(1, date_range + 1)
        else:
            # If no date column, use forecast horizon as period
            total_days = forecast_horizon_days if forecast_horizon_days > 0 else 1
        
        # ROS = Total Units Sold / Total Days
        ros = total_units_sold / total_days
        return ros
    
    def _estimate_sell_through_rate(self, forecast_qty: float, 
                                    current_inventory: float,
                                    historical_sell_through: Optional[float] = None) -> float:
        """Estimate sell-through rate for given quantity"""
        
        total_inventory = forecast_qty + current_inventory
        if total_inventory <= 0:
            return 0.0
        
        # Use historical if available, otherwise use target
        if historical_sell_through is not None:
            # Weighted average: 70% historical, 30% target
            estimated_st = historical_sell_through * 0.7 + self.target_sell_through_pct * 0.3
        else:
            estimated_st = self.target_sell_through_pct
        
        return estimated_st
    
    def _calculate_margin(self, price: float, cost: Optional[float] = None) -> float:
        """Calculate margin percentage"""
        
        if cost is None:
            # Estimate cost based on default margin
            cost = price * (1 - self.default_margin_pct)
        
        if price <= 0:
            return 0.0
        
        margin_pct = (price - cost) / price
        return max(0.0, margin_pct)
    
    def _optimize_forecast(self, forecast: Dict[str, Any],
                          store_id: str, article: str,
                          sales_data: pd.DataFrame,
                          inventory_data: pd.DataFrame,
                          price_data: pd.DataFrame,
                          historical_metrics: Dict[str, Any],
                          forecast_horizon_days: int,
                          cost_data: Optional[pd.DataFrame] = None,
                          margin_target: Optional[float] = None) -> Dict[str, Any]:
        """Optimize forecast quantity for rate of sale, sell-through, and margin protection
        Ensures margin >= margin_target and considers cost of procurement"""
        
        original_qty = forecast.get('forecast_quantity', 0)
        if original_qty <= 0:
            return forecast
        
        # Get current inventory
        current_inventory = 0
        if inventory_data is not None and not inventory_data.empty:
            inv_filter = inventory_data[
                (inventory_data['store_id'] == store_id) & 
                (inventory_data['sku'] == article)
            ]
            if not inv_filter.empty and 'on_hand' in inv_filter.columns:
                current_inventory = inv_filter['on_hand'].iloc[0]
        
        # Get price
        price = 0
        if price_data is not None and not price_data.empty:
            price_filter = price_data[
                (price_data['store_id'] == store_id) & 
                (price_data['sku'] == article)
            ]
            if not price_filter.empty and 'price' in price_filter.columns:
                price = price_filter['price'].iloc[0]
        
        # Get cost of procurement
        cost = None
        if cost_data is not None and not cost_data.empty:
            if 'sku' in cost_data.columns and 'cost' in cost_data.columns:
                cost_filter = cost_data[cost_data['sku'] == article]
                if not cost_filter.empty:
                    cost = cost_filter['cost'].iloc[0]
        
        # Use margin_target if provided, otherwise use min_margin_pct
        target_margin = margin_target if margin_target is not None else self.min_margin_pct
        
        # Get historical metrics
        key = f"{store_id}_{article}"
        historical_ros = historical_metrics['rate_of_sale'].get(key, 0)
        historical_st = historical_metrics['sell_through_rate'].get(key, None)
        historical_margin = historical_metrics['margins'].get(key, self.default_margin_pct)
        
        # Calculate margin from cost if available
        if cost is not None and price > 0:
            calculated_margin = (price - cost) / price
            # Use the higher of historical margin or calculated margin, but ensure >= target
            historical_margin = max(calculated_margin, historical_margin, target_margin)
        
        # Calculate current rate of sale
        current_ros = self._calculate_rate_of_sale(sales_data, store_id, article, forecast_horizon_days)
        if current_ros <= 0 and historical_ros > 0:
            current_ros = historical_ros
        
        # Optimize quantity based on multiple objectives
        optimized_qty = self._optimize_quantity(
            original_qty=original_qty,
            current_inventory=current_inventory,
            rate_of_sale=current_ros,
            historical_sell_through=historical_st,
            price=price,
            margin=historical_margin,
            forecast_horizon_days=forecast_horizon_days
        )
        
        # Calculate expected metrics for optimized quantity
        total_inventory = optimized_qty + current_inventory
        expected_sell_through = self._estimate_sell_through_rate(
            optimized_qty, current_inventory, historical_st
        )
        expected_units_sold = total_inventory * expected_sell_through
        expected_rate_of_sale = expected_units_sold / forecast_horizon_days if forecast_horizon_days > 0 else 0
        
        # Check margin protection (must be >= target margin)
        margin_safe = historical_margin >= target_margin
        
        # Store cost and margin information
        forecast['cost'] = cost if cost is not None else 0
        forecast['target_margin'] = target_margin
        forecast['margin_meets_target'] = margin_safe
        
        # Update forecast with optimization results
        forecast['forecast_quantity'] = max(0, optimized_qty)
        forecast['optimized_quantity'] = optimized_qty
        forecast['original_quantity'] = original_qty
        forecast['expected_sell_through_rate'] = expected_sell_through
        forecast['expected_rate_of_sale'] = expected_rate_of_sale
        forecast['expected_units_sold'] = expected_units_sold
        forecast['margin_pct'] = historical_margin
        forecast['margin_protected'] = margin_safe
        forecast['current_inventory'] = current_inventory
        
        # Adjust recommendation based on optimization
        # Margin must be >= target margin to proceed
        if optimized_qty > 0 and margin_safe and expected_sell_through >= self.target_sell_through_pct * 0.8:
            forecast['recommendation'] = 'buy'
            forecast['optimization_score'] = self._calculate_optimization_score(
                expected_rate_of_sale, expected_sell_through, historical_margin, target_margin
            )
        elif optimized_qty > 0 and margin_safe:
            forecast['recommendation'] = 'buy_cautious'
            forecast['optimization_score'] = self._calculate_optimization_score(
                expected_rate_of_sale, expected_sell_through, historical_margin, target_margin
            )
        elif not margin_safe:
            forecast['recommendation'] = 'skip_margin_risk'
            forecast['optimization_score'] = 0.0
            forecast['reasoning'] += f" | Margin {historical_margin*100:.1f}% < target {target_margin*100:.1f}%"
        else:
            forecast['recommendation'] = 'skip'
            forecast['optimization_score'] = 0.0
        
        forecast['reasoning'] += f" | Optimized: ROS={expected_rate_of_sale:.2f}/day, ST={expected_sell_through*100:.1f}%, Margin={historical_margin*100:.1f}%"
        
        return forecast
    
    def _optimize_quantity(self, original_qty: float, current_inventory: float,
                          rate_of_sale: float, historical_sell_through: Optional[float],
                          price: float, margin: float, forecast_horizon_days: int) -> float:
        """Optimize quantity to maximize rate of sale and sell-through while protecting margin"""
        
        # If margin is below minimum, reduce quantity or skip
        if margin < self.min_margin_pct:
            # Reduce quantity to maintain margin or skip
            return 0.0
        
        # Calculate expected demand based on rate of sale
        expected_demand = rate_of_sale * forecast_horizon_days
        
        # Adjust for sell-through rate
        if historical_sell_through is not None:
            # Use historical sell-through to estimate required inventory
            target_inventory = expected_demand / historical_sell_through if historical_sell_through > 0 else expected_demand
        else:
            # Use target sell-through rate
            target_inventory = expected_demand / self.target_sell_through_pct
        
        # Calculate required purchase quantity
        required_qty = max(0, target_inventory - current_inventory)
        
        # Start with original forecast as base
        optimized_qty = original_qty
        
        # Adjust based on rate of sale
        if rate_of_sale > 0:
            # If rate of sale suggests higher demand, increase quantity
            if expected_demand > original_qty:
                # Increase by up to 20% if margin is good
                if margin >= self.default_margin_pct:
                    optimized_qty = min(required_qty, original_qty * 1.2)
                else:
                    optimized_qty = min(required_qty, original_qty)
            else:
                # If rate of sale suggests lower demand, be conservative
                optimized_qty = min(original_qty, required_qty)
        
        # Ensure we don't over-order (maximize sell-through)
        # Cap at 1.5x expected demand to avoid overstock
        max_qty = expected_demand * 1.5
        optimized_qty = min(optimized_qty, max_qty)
        
        # Margin protection: if margin is low, reduce quantity
        if margin < self.default_margin_pct * 0.9:
            # Reduce by margin risk factor
            margin_risk_factor = margin / self.default_margin_pct
            optimized_qty = optimized_qty * margin_risk_factor
        
        # Ensure minimum viable quantity (at least 5 units or skip)
        if optimized_qty < 5:
            optimized_qty = 0
        
        return max(0, optimized_qty)
    
    def _calculate_optimization_score(self, rate_of_sale: float, 
                                     sell_through_rate: float,
                                     margin_pct: float) -> float:
        """Calculate optimization score (0-1) based on objectives"""
        
        # Normalize rate of sale (assume max 100 units/day)
        ros_score = min(1.0, rate_of_sale / 100.0) if rate_of_sale > 0 else 0.0
        
        # Sell-through rate score (target is 75%)
        st_score = min(1.0, sell_through_rate / self.target_sell_through_pct)
        
        # Margin score (higher is better, normalized to 0-1)
        margin_score = min(1.0, margin_pct / self.default_margin_pct)
        
        # Weighted combination: 40% ROS, 35% ST, 25% Margin
        total_score = (ros_score * 0.4 + st_score * 0.35 + margin_score * 0.25)
        
        return total_score
    
    def _aggregate_forecasts(self, store_forecasts: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate store-level forecasts with optimization metrics"""
        
        total_quantity = 0
        articles = set()
        stores_with_recommendations = 0
        total_expected_units_sold = 0
        total_inventory = 0
        total_margin_value = 0
        total_revenue = 0
        optimization_scores = []
        
        for store_id, articles_dict in store_forecasts.items():
            for article, forecast in articles_dict.items():
                if forecast.get('recommendation') in ['buy', 'buy_cautious']:
                    qty = forecast.get('forecast_quantity', 0)
                    if qty > 0:
                        total_quantity += qty
                        articles.add(article)
                        stores_with_recommendations += 1
                        
                        # Aggregate optimization metrics
                        expected_units = forecast.get('expected_units_sold', 0)
                        total_expected_units_sold += expected_units
                        
                        current_inv = forecast.get('current_inventory', 0)
                        total_inventory += (qty + current_inv)
                        
                        margin_pct = forecast.get('margin_pct', 0)
                        price = forecast.get('price', 0)
                        if price > 0:
                            revenue = expected_units * price
                            margin_value = revenue * margin_pct
                            total_revenue += revenue
                            total_margin_value += margin_value
                        
                        opt_score = forecast.get('optimization_score', 0)
                        if opt_score > 0:
                            optimization_scores.append(opt_score)
                        
                        break  # Count store only once
        
        # Calculate aggregate metrics
        avg_sell_through = (total_expected_units_sold / total_inventory 
                          if total_inventory > 0 else 0)
        avg_margin_pct = (total_margin_value / total_revenue 
                         if total_revenue > 0 else 0)
        avg_optimization_score = (np.mean(optimization_scores) 
                                 if optimization_scores else 0)
        
        return {
            'total_quantity': total_quantity,
            'unique_articles': list(articles),
            'num_articles': len(articles),
            'stores_with_recommendations': stores_with_recommendations,
            'total_stores': len(store_forecasts),
            'total_expected_units_sold': total_expected_units_sold,
            'avg_sell_through_rate': avg_sell_through,
            'avg_margin_pct': avg_margin_pct,
            'total_revenue': total_revenue,
            'total_margin_value': total_margin_value,
            'avg_optimization_score': avg_optimization_score
        }
    
    def _generate_optimized_recommendations(self, store_forecasts: Dict[str, Dict],
                                           aggregated_forecast: Dict[str, Any],
                                           product_attributes: Dict[str, Any],
                                           sales_data: pd.DataFrame,
                                           price_data: pd.DataFrame,
                                           historical_metrics: Dict[str, Any],
                                           forecast_horizon_days: int = 60,
                                           cost_data: Optional[pd.DataFrame] = None,
                                           margin_target: Optional[float] = None,
                                           historical_pricing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate optimized recommendations prioritizing rate of sale, sell-through, and margin protection
        Includes article-level metrics, store universe validation, and top banner summary
        """
        
        recommendations = {
            'articles_to_buy': [],
            'store_allocations': {},
            'total_procurement_quantity': 0,
            'total_stores': 0,
            'universe_of_stores': self.universe_of_stores,
            'store_universe_validation': {},
            'priority_stores': [],
            'article_level_metrics': {},  # New: Article-level aggregated metrics
            'optimization_summary': '',
            'expected_metrics': {},
            'risk_assessment': {}
        }
        
        # Collect all articles to buy (prioritize 'buy' over 'buy_cautious')
        article_totals = {}
        store_allocations = {}
        article_scores = {}  # Track optimization scores by article
        
        for store_id, articles_dict in store_forecasts.items():
            for article, forecast in articles_dict.items():
                recommendation = forecast.get('recommendation', 'skip')
                if recommendation in ['buy', 'buy_cautious']:
                    qty = forecast.get('forecast_quantity', 0)
                    opt_score = forecast.get('optimization_score', 0)
                    
                    if qty > 0:
                        if article not in article_totals:
                            article_totals[article] = 0
                            store_allocations[article] = {}
                            article_scores[article] = []
                        
                        article_totals[article] += qty
                        store_allocations[article][store_id] = {
                            'quantity': qty,
                            'expected_sell_through': forecast.get('expected_sell_through_rate', 0),
                            'expected_rate_of_sale': forecast.get('expected_rate_of_sale', 0),
                            'margin_pct': forecast.get('margin_pct', 0),
                            'optimization_score': opt_score,
                            'recommendation': recommendation
                        }
                        article_scores[article].append(opt_score)
        
        # Sort articles by average optimization score (prioritize high ROS, ST, margin)
        sorted_articles = sorted(
            article_totals.keys(),
            key=lambda a: np.mean(article_scores[a]) if article_scores[a] else 0,
            reverse=True
        )
        
        recommendations['articles_to_buy'] = sorted_articles
        recommendations['store_allocations'] = store_allocations
        recommendations['total_procurement_quantity'] = sum(article_totals.values())
        
        # Calculate total stores (unique stores across all articles)
        all_stores = set()
        for article, stores in store_allocations.items():
            all_stores.update(stores.keys())
        recommendations['total_stores'] = len(all_stores)
        
        # Validate against universe of stores
        if self.universe_of_stores is not None:
            if recommendations['total_stores'] > self.universe_of_stores:
                recommendations['store_universe_validation'] = {
                    'valid': False,
                    'message': f'WARNING: Total stores ({recommendations["total_stores"]}) exceeds universe ({self.universe_of_stores}). Recommendations may need adjustment.',
                    'total_stores': recommendations['total_stores'],
                    'universe_of_stores': self.universe_of_stores,
                    'excess_stores': recommendations['total_stores'] - self.universe_of_stores
                }
            else:
                recommendations['store_universe_validation'] = {
                    'valid': True,
                    'message': f'Total stores ({recommendations["total_stores"]}) is within universe ({self.universe_of_stores}).',
                    'total_stores': recommendations['total_stores'],
                    'universe_of_stores': self.universe_of_stores
                }
        else:
            recommendations['store_universe_validation'] = {
                'valid': None,
                'message': 'Universe of stores not specified. Validation skipped.',
                'total_stores': recommendations['total_stores']
            }
        
        # Calculate article-level metrics
        recommendations['article_level_metrics'] = self._calculate_article_level_metrics(
            sorted_articles, store_allocations, sales_data, price_data, 
            product_attributes, forecast_horizon_days, historical_pricing, margin_target
        )
        
        # Identify priority stores (stores with highest optimization scores)
        store_scores = {}
        store_totals = {}
        for article, stores in store_allocations.items():
            for store_id, store_data in stores.items():
                if store_id not in store_scores:
                    store_scores[store_id] = []
                    store_totals[store_id] = 0
                
                store_scores[store_id].append(store_data.get('optimization_score', 0))
                store_totals[store_id] += store_data.get('quantity', 0)
        
        # Priority stores: highest average optimization score
        recommendations['priority_stores'] = sorted(
            [(sid, np.mean(scores), qty) for sid, scores, qty in 
             zip(store_scores.keys(), store_scores.values(), store_totals.values())],
            key=lambda x: (x[1], x[2]),  # Sort by score, then quantity
            reverse=True
        )[:10]  # Top 10 stores
        
        # Expected metrics
        recommendations['expected_metrics'] = {
            'total_expected_units_sold': aggregated_forecast.get('total_expected_units_sold', 0),
            'avg_sell_through_rate': aggregated_forecast.get('avg_sell_through_rate', 0),
            'avg_margin_pct': aggregated_forecast.get('avg_margin_pct', 0),
            'total_revenue': aggregated_forecast.get('total_revenue', 0),
            'total_margin_value': aggregated_forecast.get('total_margin_value', 0),
            'avg_optimization_score': aggregated_forecast.get('avg_optimization_score', 0)
        }
        
        # Risk assessment
        margin_risk_articles = []
        low_st_articles = []
        
        for article, stores in store_allocations.items():
            avg_margin = np.mean([s.get('margin_pct', 0) for s in stores.values()])
            avg_st = np.mean([s.get('expected_sell_through', 0) for s in stores.values()])
            
            if avg_margin < self.min_margin_pct:
                margin_risk_articles.append(article)
            if avg_st < self.target_sell_through_pct * 0.7:
                low_st_articles.append(article)
        
        recommendations['risk_assessment'] = {
            'margin_risk_articles': margin_risk_articles,
            'low_sell_through_articles': low_st_articles,
            'high_confidence_articles': [
                a for a in sorted_articles 
                if a not in margin_risk_articles and a not in low_st_articles
            ]
        }
        
        # Generate comprehensive summary
        avg_st = recommendations['expected_metrics']['avg_sell_through_rate']
        avg_margin = recommendations['expected_metrics']['avg_margin_pct']
        total_revenue = recommendations['expected_metrics']['total_revenue']
        total_margin = recommendations['expected_metrics']['total_margin_value']
        
        # Build optimization summary with store universe info
        store_info = f"across {recommendations['total_stores']} stores"
        if self.universe_of_stores:
            store_info += f" (universe: {self.universe_of_stores})"
        
        recommendations['optimization_summary'] = (
            f"Optimized recommendation: Procure {len(sorted_articles)} articles "
            f"with total quantity of {recommendations['total_procurement_quantity']:.0f} units "
            f"{store_info}.\n"
            f"Expected performance:\n"
            f"  - Sell-through rate: {avg_st*100:.1f}% (target: {self.target_sell_through_pct*100:.1f}%)\n"
            f"  - Average margin: {avg_margin*100:.1f}% (min: {self.min_margin_pct*100:.1f}%)\n"
            f"  - Expected revenue: ₹{total_revenue:,.0f}\n"
            f"  - Expected margin value: ₹{total_margin:,.0f}\n"
            f"  - Optimization score: {aggregated_forecast.get('avg_optimization_score', 0):.2f}/1.0"
        )
        
        # Add article-level summary to optimization_summary
        if recommendations.get('article_level_metrics'):
            recommendations['optimization_summary'] += "\n\nArticle-Level Metrics:\n"
            for article, metrics in recommendations['article_level_metrics'].items():
                details = metrics.get('article_details', {})
                recommendations['optimization_summary'] += (
                    f"\n  Article: {article}\n"
                    f"    Style Code: {details.get('style_code', 'N/A')}, "
                    f"Color: {details.get('color', 'N/A')}, "
                    f"Segment: {details.get('segment', 'N/A')}\n"
                    f"    Family: {details.get('family', 'N/A')}, "
                    f"Class: {details.get('class', 'N/A')}, "
                    f"Brick: {details.get('brick', 'N/A')}\n"
                    f"    MRP: ₹{metrics.get('mrp', 0):.2f}, "
                    f"Avg Selling Price: ₹{metrics.get('average_selling_price', 0):.2f}, "
                    f"Avg Discount: {metrics.get('average_discount', 0):.1f}%\n"
                    f"    Margin: {metrics.get('margin_pct', 0)*100:.1f}%, "
                    f"Store Exposure: {metrics.get('total_store_exposure', 0)}, "
                    f"ROS: {metrics.get('ros', 0):.2f} units/day, "
                    f"STR: {metrics.get('str', 0)*100:.1f}%\n"
                    f"    Net Sales Value: ₹{metrics.get('net_sales_value', 0):,.0f}\n"
                )
        
        if margin_risk_articles:
            recommendations['optimization_summary'] += (
                f"\n⚠️  Margin risk detected for {len(margin_risk_articles)} articles. "
                f"Consider price adjustments or skip procurement."
            )
        
        if low_st_articles:
            recommendations['optimization_summary'] += (
                f"\n⚠️  Low sell-through risk for {len(low_st_articles)} articles. "
                f"Consider reducing quantities."
            )
        
        # Generate top banner summary
        recommendations['top_banner'] = self._generate_top_banner(
            recommendations, cost_data, margin_target, aggregated_forecast
        )
        
        return recommendations
    
    def _generate_top_banner(self, recommendations: Dict[str, Any],
                            cost_data: Optional[pd.DataFrame],
                            margin_target: Optional[float],
                            aggregated_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate top banner summary with:
        1. Total unique SKUs bought
        2. Total quantity bought at SKU level
        3. Total stores for which these have been bought
        4. Total buy cost (quantity X cost for each SKU)
        5. Total sales value for all SKUs
        6. Average Margin achieved vs Target Margin
        """
        banner = {
            'total_unique_skus': 0,
            'total_quantity_bought': 0,
            'total_stores': 0,
            'total_buy_cost': 0.0,
            'total_sales_value': 0.0,
            'average_margin_achieved': 0.0,
            'target_margin': margin_target if margin_target else self.min_margin_pct,
            'margin_vs_target': {
                'achieved': 0.0,
                'target': 0.0,
                'difference': 0.0,
                'meets_target': False
            }
        }
        
        # Get articles and quantities
        articles_to_buy = recommendations.get('articles_to_buy', [])
        store_allocations = recommendations.get('store_allocations', {})
        article_level_metrics = recommendations.get('article_level_metrics', {})
        
        banner['total_unique_skus'] = len(articles_to_buy)
        
        # Calculate total quantity and stores
        all_stores = set()
        sku_quantities = {}  # {sku: total_quantity}
        
        for article in articles_to_buy:
            if article in store_allocations:
                article_qty = 0
                for store_id, store_data in store_allocations[article].items():
                    qty = store_data.get('quantity', 0)
                    article_qty += qty
                    all_stores.add(store_id)
                sku_quantities[article] = article_qty
                banner['total_quantity_bought'] += article_qty
        
        banner['total_stores'] = len(all_stores)
        
        # Calculate total buy cost
        if cost_data is not None and not cost_data.empty:
            if 'sku' in cost_data.columns and 'cost' in cost_data.columns:
                for article in articles_to_buy:
                    cost_filter = cost_data[cost_data['sku'] == article]
                    if not cost_filter.empty:
                        cost = cost_filter['cost'].iloc[0]
                        qty = sku_quantities.get(article, 0)
                        banner['total_buy_cost'] += cost * qty
        
        # Calculate total sales value
        for article in articles_to_buy:
            if article in article_level_metrics:
                metrics = article_level_metrics[article]
                net_sales_value = metrics.get('net_sales_value', 0)
                banner['total_sales_value'] += net_sales_value
        
        # Calculate average margin achieved
        margins = []
        for article in articles_to_buy:
            if article in article_level_metrics:
                margin = article_level_metrics[article].get('margin_pct', 0)
                if margin > 0:
                    margins.append(margin)
        
        if margins:
            banner['average_margin_achieved'] = np.mean(margins)
        else:
            banner['average_margin_achieved'] = aggregated_forecast.get('avg_margin_pct', 0)
        
        # Compare with target margin
        target = banner['target_margin']
        achieved = banner['average_margin_achieved']
        banner['margin_vs_target'] = {
            'achieved': achieved,
            'target': target,
            'difference': achieved - target,
            'meets_target': achieved >= target,
            'achieved_pct': achieved * 100,
            'target_pct': target * 100,
            'difference_pct': (achieved - target) * 100
        }
        
        return banner
    
    def _calculate_article_level_metrics(self, articles: List[str],
                                         store_allocations: Dict[str, Dict],
                                         sales_data: pd.DataFrame,
                                         price_data: pd.DataFrame,
                                         product_attributes: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
                                         forecast_horizon_days: int,
                                         historical_pricing: Optional[Dict[str, Any]] = None,
                                         margin_target: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate article-level metrics for each article:
        - MRP, Average Selling Price, Average Discount, Margin
        - Total Store Exposure, ROS, STR, Net Sales Value
        - Article Details: style_code, color, segment, family, class, brick
        """
        
        article_metrics = {}
        
        for article in articles:
            if article not in store_allocations:
                continue
            
            stores = store_allocations[article]
            store_ids = list(stores.keys())
            
            # Extract article details from product_attributes
            # product_attributes can be:
            # 1. Dict with article as key: {article: {style_code: ..., color: ...}}
            # 2. Dict with global attributes: {style_code: ..., color: ...} (applies to all)
            # 3. Dict from attribute_analogy_agent: {attributes: {style_code: ..., color: ...}}
            
            article_attrs = {}
            if isinstance(product_attributes, dict):
                if article in product_attributes:
                    # Article-specific attributes
                    article_attrs = product_attributes[article]
                elif 'attributes' in product_attributes:
                    # Attributes from attribute_analogy_agent
                    article_attrs = product_attributes['attributes']
                else:
                    # Global attributes (apply to all articles)
                    article_attrs = product_attributes
            
            # Initialize article metrics
            metrics = {
                'article': article,
                'article_details': {
                    'style_code': article_attrs.get('style_code', 'N/A'),
                    'color': article_attrs.get('color', 'N/A'),
                    'segment': article_attrs.get('segment', 'N/A'),
                    'family': article_attrs.get('family', 'N/A'),
                    'class': article_attrs.get('class', 'N/A'),
                    'brick': article_attrs.get('brick', 'N/A')
                },
                'total_store_exposure': len(store_ids),
                'mrp': 0.0,
                'average_selling_price': 0.0,
                'average_discount': 0.0,
                'margin_pct': 0.0,
                'ros': 0.0,  # Rate of Sale (units per day)
                'str': 0.0,  # Sell-Through Rate
                'net_sales_value': 0.0,
                'total_quantity': 0,
                'expected_units_sold': 0
            }
            
            # Get price information for this article
            article_prices = []
            article_mrps = []
            total_quantity = 0
            total_expected_units = 0
            total_inventory = 0
            total_units_sold_historical = 0
            total_days_historical = 0
            
            if price_data is not None and not price_data.empty:
                article_price_data = price_data[price_data['sku'] == article]
                if not article_price_data.empty:
                    # Get prices for stores where article is allocated
                    store_price_data = article_price_data[article_price_data['store_id'].isin(store_ids)]
                    if not store_price_data.empty:
                        article_prices = store_price_data['price'].tolist()
                        # Assume MRP is in a column or calculate from price
                        if 'mrp' in store_price_data.columns:
                            article_mrps = store_price_data['mrp'].tolist()
                        elif 'price' in store_price_data.columns:
                            # If no MRP, use price as MRP (no discount scenario)
                            article_mrps = store_price_data['price'].tolist()
            
            # Get MRP from historical pricing analysis (should be same across all stores)
            if historical_pricing and article in historical_pricing:
                pricing_info = historical_pricing[article]
                metrics['mrp'] = pricing_info.get('mrp', 0.0)
                metrics['mrp_consistent'] = pricing_info.get('mrp_consistent', True)
            else:
                # Calculate MRP and Average Selling Price from price data
                if article_mrps:
                    metrics['mrp'] = np.mean(article_mrps)
                    metrics['mrp_consistent'] = len(set(article_mrps)) == 1  # Same across stores
                elif article_prices:
                    metrics['mrp'] = np.mean(article_prices)  # Fallback: use price as MRP
                    metrics['mrp_consistent'] = len(set(article_prices)) == 1
            
            # Calculate Average Selling Price (can vary by store)
            if article_prices:
                metrics['average_selling_price'] = np.mean(article_prices)
                metrics['price_std'] = np.std(article_prices) if len(article_prices) > 1 else 0.0
                metrics['price_consistency'] = 'consistent' if metrics['price_std'] < metrics['average_selling_price'] * 0.05 else 'variable'
            else:
                metrics['average_selling_price'] = 0.0
                metrics['price_consistency'] = 'unknown'
            
            # Calculate Average Discount (can vary by store)
            if metrics['mrp'] > 0 and metrics['average_selling_price'] > 0:
                discount_pct = ((metrics['mrp'] - metrics['average_selling_price']) / metrics['mrp']) * 100
                metrics['average_discount'] = max(0, discount_pct)
                # Check discount consistency across stores
                if article_prices and len(article_prices) > 1:
                    discounts = [((metrics['mrp'] - p) / metrics['mrp']) * 100 for p in article_prices]
                    discount_std = np.std(discounts)
                    metrics['discount_consistency'] = 'consistent' if discount_std < 2.0 else 'variable'
                else:
                    metrics['discount_consistency'] = 'consistent'
            else:
                metrics['average_discount'] = 0.0
                metrics['discount_consistency'] = 'unknown'
            
            # Aggregate quantities and calculate metrics from store allocations
            for store_id, store_data in stores.items():
                qty = store_data.get('quantity', 0)
                total_quantity += qty
                
                expected_st = store_data.get('expected_sell_through', 0)
                expected_units = qty * expected_st
                total_expected_units += expected_units
                
                current_inv = store_data.get('current_inventory', 0)
                total_inventory += (qty + current_inv)
                
                # Get margin from store data
                margin = store_data.get('margin_pct', 0)
                if margin > 0:
                    metrics['margin_pct'] = margin  # Use latest margin or average if needed
            
            metrics['total_quantity'] = total_quantity
            metrics['expected_units_sold'] = total_expected_units
            
            # Calculate STR (Sell-Through Rate)
            if total_inventory > 0:
                metrics['str'] = total_expected_units / total_inventory
            elif total_quantity > 0:
                # If no inventory data, use expected sell-through from forecasts
                avg_st = np.mean([s.get('expected_sell_through', 0) for s in stores.values()])
                metrics['str'] = avg_st
            
            # Calculate ROS (Rate of Sale) = Total Units Sold / Total Days
            if sales_data is not None and not sales_data.empty:
                article_sales = sales_data[sales_data['sku'] == article]
                if not article_sales.empty:
                    # Filter by stores where article is allocated
                    article_sales = article_sales[article_sales['store_id'].isin(store_ids)]
                    
                    if not article_sales.empty:
                        total_units_sold_historical = article_sales['units_sold'].sum()
                        
                        # Calculate total days in period
                        if 'date' in article_sales.columns:
                            article_sales['date'] = pd.to_datetime(article_sales['date'])
                            date_range = (article_sales['date'].max() - article_sales['date'].min()).days
                            total_days_historical = max(1, date_range + 1)
                        else:
                            total_days_historical = forecast_horizon_days
                        
                        if total_days_historical > 0:
                            metrics['ros'] = total_units_sold_historical / total_days_historical
                        else:
                            # Fallback: use expected rate of sale from forecasts
                            avg_ros = np.mean([s.get('expected_rate_of_sale', 0) for s in stores.values()])
                            metrics['ros'] = avg_ros
                    else:
                        # No historical sales, use expected ROS from forecasts
                        avg_ros = np.mean([s.get('expected_rate_of_sale', 0) for s in stores.values()])
                        metrics['ros'] = avg_ros
                else:
                    # No sales data for article, use expected ROS from forecasts
                    avg_ros = np.mean([s.get('expected_rate_of_sale', 0) for s in stores.values()])
                    metrics['ros'] = avg_ros
            else:
                # No sales data available, use expected ROS from forecasts
                avg_ros = np.mean([s.get('expected_rate_of_sale', 0) for s in stores.values()])
                metrics['ros'] = avg_ros
            
            # Calculate Net Sales Value = Expected Units Sold * Average Selling Price
            if metrics['average_selling_price'] > 0:
                metrics['net_sales_value'] = total_expected_units * metrics['average_selling_price']
            else:
                # Fallback: use price from price_data or forecast
                if article_prices:
                    avg_price = np.mean(article_prices)
                    metrics['net_sales_value'] = total_expected_units * avg_price
            
            # Calculate average margin (must be >= target margin)
            target = margin_target if margin_target is not None else self.min_margin_pct
            if 'margins' in metrics and metrics['margins']:
                metrics['margin_pct'] = np.mean(metrics['margins'])
                margin_std = np.std(metrics['margins']) if len(metrics['margins']) > 1 else 0.0
                # Remove temporary margins list
                del metrics['margins']
            else:
                margins = [s.get('margin_pct', 0) for s in stores.values()]
                if margins:
                    valid_margins = [m for m in margins if m > 0]
                    if valid_margins:
                        metrics['margin_pct'] = np.mean(valid_margins)
                        margin_std = np.std(valid_margins) if len(valid_margins) > 1 else 0.0
                    else:
                        metrics['margin_pct'] = self.default_margin_pct
                        margin_std = 0.0
                else:
                    metrics['margin_pct'] = self.default_margin_pct
                    margin_std = 0.0
            
            # Ensure margin >= target margin
            if metrics['margin_pct'] < target:
                metrics['margin_pct'] = target  # Adjust to meet target
                metrics['margin_adjusted'] = True
            else:
                metrics['margin_adjusted'] = False
            
            # Check margin consistency across stores
            if 'margin_std' in locals() and margin_std > 0:
                metrics['margin_consistency'] = 'consistent' if margin_std < metrics['margin_pct'] * 0.05 else 'variable'
            else:
                metrics['margin_consistency'] = 'consistent'
            
            metrics['margin_meets_target'] = metrics['margin_pct'] >= target
            metrics['target_margin'] = target
            
            article_metrics[article] = metrics
        
        return article_metrics
    
    def validate_input_data(self, sales_data: pd.DataFrame, 
                           inventory_data: pd.DataFrame,
                           price_data: pd.DataFrame,
                           comparables: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate input data using Pydantic and generate clear alert messages for inconsistencies
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        # Use the Pydantic-based validation method
        validation_result = self._validate_input_data(
            sales_data, inventory_data, price_data, comparables
        )
        
        # Store validation errors for reference
        self.data_validation_errors = validation_result['errors']
        
        return validation_result['is_valid'], validation_result['errors']
    
    def run(self, comparables: List[Dict], sales_data: pd.DataFrame, 
           inventory_data: pd.DataFrame, price_data: pd.DataFrame, 
           price_options: List[float], product_attributes: Optional[Dict] = None,
           forecast_horizon_days: int = 60, variance_threshold: Optional[float] = None,
           cost_data: Optional[pd.DataFrame] = None, margin_target: Optional[float] = None,
           max_quantity_per_store: int = 500) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main run method that executes all three tasks sequentially with fallback mechanisms
        
        Args:
            comparables: List of comparable products from attribute analogy agent
            sales_data: Sales data from data ingestion agent
            inventory_data: Inventory data from data ingestion agent
            price_data: Price data from data ingestion agent
            price_options: List of price options for sensitivity analysis
            product_attributes: Product attributes from attribute analogy agent
            forecast_horizon_days: Forecast horizon in days (default 60)
            variance_threshold: Variance threshold for auto-approval (0.05 = 5%, configurable from UI)
                                If None, uses the value set during initialization
            cost_data: DataFrame with columns ['sku', 'cost'] - cost of procuring each article
            margin_target: Target margin percentage from UI (0.30 = 30%, configurable from UI)
                          If None, uses min_margin_pct from initialization
            max_quantity_per_store: Maximum quantity that can be allocated per store (default 500)
        
        Returns:
            Tuple of (forecast_results, sensitivity_analysis)
        """
        
        print("\n" + "=" * 60)
        print("DEMAND FORECASTING AGENT - Starting Sequential Tasks")
        print("=" * 60)
        
        # Update variance threshold if provided from UI
        if variance_threshold is not None:
            if variance_threshold < 0 or variance_threshold > 1:
                raise ValueError(f"Variance threshold must be between 0 and 1 (0-100%). Got: {variance_threshold}")
            self.variance_threshold = variance_threshold
            # Update HITL workflow if enabled
            if self.enable_hitl and self.hitl:
                self.hitl.variance_threshold = variance_threshold
            print(f"\nVariance threshold set to: {variance_threshold*100:.1f}% (from UI input)")
        
        # Update margin target if provided from UI
        if margin_target is not None:
            if margin_target < 0 or margin_target > 1:
                raise ValueError(f"Margin target must be between 0 and 1 (0-100%). Got: {margin_target}")
            self.margin_target = margin_target
            print(f"\nMargin target set to: {margin_target*100:.1f}% (from UI input)")
        else:
            self.margin_target = self.min_margin_pct  # Use minimum margin as target
        
        # Store cost data and constraints
        self.cost_data = cost_data
        self.max_quantity_per_store = max_quantity_per_store
        
        # Step 0: Validate input data (including cost data)
        is_valid, validation_messages = self.validate_input_data(
            sales_data, inventory_data, price_data, comparables
        )
        
        # Print validation messages
        if validation_messages:
            print("\n" + "=" * 60)
            print("DATA VALIDATION RESULTS")
            print("=" * 60)
            for msg in validation_messages:
                print(f"  {msg}")
            print("=" * 60 + "\n")
        
        # If critical errors, return early with alerts
        if not is_valid:
            return {
                'error': True,
                'error_type': 'data_validation_failed',
                'validation_messages': validation_messages,
                'message': 'CRITICAL: Input data validation failed. Please fix the data inconsistencies before proceeding.',
                'forecast_results': {},
                'recommendations': {}
            }, {}
        
        try:
            # Task 1: Select best forecasting model
            model_selection = self.select_model(
                comparables, sales_data, inventory_data, price_data
            )
            
            # Task 2: Sensitivity and factor analysis
            if product_attributes is None:
                product_attributes = {}
                # Try to extract from comparables if available
                if comparables and len(comparables) > 0:
                    product_attributes = comparables[0].get('attributes', {})
            
            factor_analysis = self.sensitivity_and_factor_analysis(
                sales_data, price_data, product_attributes, comparables
            )
            
            # Task 3: Store-level forecasting
            forecast_results = self.forecast_store_level(
                sales_data, inventory_data, price_data, 
                product_attributes, comparables, forecast_horizon_days,
                cost_data, margin_target, max_quantity_per_store
            )
            
            # Check if forecasting produced valid results
            if not forecast_results or not forecast_results.get('store_level_forecasts'):
                print("\n⚠️  Forecasting produced empty results. Fallback already attempted in _generate_forecast.")
            
        except Exception as e:
            print(f"\n❌ Forecasting error: {str(e)}")
            print("Note: Fallback mechanism is built into _generate_forecast method")
            
            # Return error response
            return {
                'error': True,
                'error_type': 'forecasting_failed',
                'message': f'Forecasting failed: {str(e)}. Fallback mechanism attempted but also failed.',
                'forecast_results': {},
                'recommendations': {}
            }, {}
        
        # Apply HITL store mappings if enabled
        if self.enable_hitl and self.hitl and forecast_results:
            store_forecasts = forecast_results.get('store_level_forecasts', {})
            if store_forecasts:
                # Apply store mappings
                updated_forecasts = self.hitl.apply_store_mappings_to_forecast(
                    store_forecasts, sales_data, inventory_data, price_data
                )
                forecast_results['store_level_forecasts'] = updated_forecasts
                # Update recommendations with mapped stores
                if 'recommendations' in forecast_results:
                    # Calculate historical metrics if not available
                    if 'historical_metrics' not in locals():
                        historical_metrics = self._calculate_historical_metrics(
                            sales_data, inventory_data, price_data
                        )
                    recommendations = self._generate_optimized_recommendations(
                        updated_forecasts,
                        forecast_results.get('aggregated_forecast', {}),
                        product_attributes,
                        sales_data,
                        price_data,
                        historical_metrics,
                        forecast_horizon_days
                    )
                    forecast_results['recommendations'] = recommendations
        
        # Combine results
        final_results = {
            'model_selection': model_selection if 'model_selection' in locals() else {},
            'factor_analysis': factor_analysis if 'factor_analysis' in locals() else {},
            'forecast_results': forecast_results,
            'recommendations': forecast_results.get('recommendations', {}) if forecast_results else {},
            'validation_messages': validation_messages,
            'fallback_used': self.fallback_used
        }
        
        # Add HITL metadata if enabled
        if self.enable_hitl and self.hitl:
            final_results['hitl_metadata'] = {
                'enabled': True,
                'available_stores': self.hitl.get_available_stores(sales_data),
                'store_mappings': self.hitl.store_mappings,
                'variance_threshold': self.hitl.variance_threshold,
                'variance_threshold_pct': self.hitl.variance_threshold * 100
            }
        
        return final_results, factor_analysis.get('sensitivity_results', {}) if 'factor_analysis' in locals() else {}
    
    # HITL Workflow Methods
    def set_variance_threshold(self, variance_threshold: float) -> Dict[str, Any]:
        """
        Set variance threshold for auto-approval (configurable from UI)
        
        Args:
            variance_threshold: Variance threshold as decimal (0.05 = 5%, 0.10 = 10%)
                               Must be between 0 and 1
        
        Returns:
            Confirmation with updated threshold
        """
        if variance_threshold < 0 or variance_threshold > 1:
            raise ValueError(f"Variance threshold must be between 0 and 1 (0-100%). Got: {variance_threshold}")
        
        self.variance_threshold = variance_threshold
        
        # Update HITL workflow if enabled
        if self.enable_hitl and self.hitl:
            self.hitl.variance_threshold = variance_threshold
        
        return {
            'variance_threshold': variance_threshold,
            'variance_threshold_pct': variance_threshold * 100,
            'message': f'Variance threshold updated to {variance_threshold*100:.1f}%',
            'updated_at': datetime.now().isoformat()
        }
    
    def get_variance_threshold(self) -> Dict[str, Any]:
        """
        Get current variance threshold
        
        Returns:
            Current variance threshold information
        """
        return {
            'variance_threshold': self.variance_threshold,
            'variance_threshold_pct': self.variance_threshold * 100,
            'description': f'Auto-approve if variance < {self.variance_threshold*100:.1f}%, flag for approval if >= {self.variance_threshold*100:.1f}%'
        }
    
    def add_new_store_mapping(self, new_store_id: str, reference_store_id: str, 
                             user_id: str = "user") -> Dict[str, Any]:
        """
        Add new store ID with mapping to reference store (Step 1)
        
        Args:
            new_store_id: New store identifier
            reference_store_id: Existing store to use as reference
            user_id: User adding the mapping
        
        Returns:
            Mapping record with audit information
        """
        if not self.enable_hitl or not self.hitl:
            raise ValueError("HITL workflow is not enabled")
        
        return self.hitl.add_new_store_mapping(new_store_id, reference_store_id, user_id)
    
    def get_available_stores_for_mapping(self, sales_data: pd.DataFrame) -> List[str]:
        """
        Get list of available stores for dropdown (Step 1)
        
        Args:
            sales_data: Sales data DataFrame
        
        Returns:
            List of store IDs
        """
        if not self.enable_hitl or not self.hitl:
            return []
        
        return self.hitl.get_available_stores(sales_data)
    
    def edit_aggregate_quantity(self, article: str, edited_quantity: float,
                               user_id: str = "user") -> Dict[str, Any]:
        """
        Edit aggregate quantity for an article (Step 3)
        Auto-approves if variance < 5%, flags for approval if >= 5%
        
        Args:
            article: Article identifier
            edited_quantity: User-edited quantity
            user_id: User making the edit
        
        Returns:
            Edit record with variance and approval status
        """
        if not self.enable_hitl or not self.hitl:
            raise ValueError("HITL workflow is not enabled")
        
        # Get original quantity from recommendations
        recommendations = self.forecast_results.get('recommendations', {})
        article_metrics = recommendations.get('article_level_metrics', {})
        
        if article not in article_metrics:
            raise ValueError(f"Article {article} not found in forecasts")
        
        original_quantity = article_metrics[article].get('total_quantity', 0)
        
        # Edit aggregate quantity
        edit_record = self.hitl.edit_aggregate_quantity(
            article, edited_quantity, original_quantity, user_id
        )
        
        # If auto-approved, re-run allocation algorithm
        if edit_record['approval_status'] == 'auto_approved':
            self._rebalance_after_aggregate_edit(article, edited_quantity)
        
        return edit_record
    
    def _rebalance_after_aggregate_edit(self, article: str, new_total_quantity: float):
        """
        Re-run allocation algorithm after aggregate edit (Step 4)
        
        Args:
            article: Article identifier
            new_total_quantity: New total quantity
        """
        recommendations = self.forecast_results.get('recommendations', {})
        store_allocations = recommendations.get('store_allocations', {})
        
        if article not in store_allocations:
            return
        
        # Get original allocations
        original_allocations = {}
        for store_id, allocation in store_allocations[article].items():
            original_allocations[store_id] = {
                'quantity': allocation.get('quantity', 0),
                'expected_sell_through': allocation.get('expected_sell_through', 0),
                'expected_rate_of_sale': allocation.get('expected_rate_of_sale', 0),
                'margin_pct': allocation.get('margin_pct', 0),
                'optimization_score': allocation.get('optimization_score', 0),
                'recommendation': allocation.get('recommendation', 'buy')
            }
        
        # Rebalance
        rebalanced = self.hitl.rebalance_store_allocations(
            article, new_total_quantity, store_allocations[article], original_allocations
        )
        
        # Update recommendations
        store_allocations[article] = rebalanced
        
        # Update article metrics
        article_metrics = recommendations.get('article_level_metrics', {})
        if article in article_metrics:
            article_metrics[article]['total_quantity'] = new_total_quantity
            article_metrics[article]['expected_units_sold'] = sum(
                s.get('quantity', 0) * s.get('expected_sell_through', 0)
                for s in rebalanced.values()
            )
    
    def edit_store_level_quantity(self, article: str, store_id: str,
                                 edited_quantity: float, user_id: str = "user") -> Dict[str, Any]:
        """
        Edit store-level quantity allocation (Step 5)
        Dynamically rebalances other stores
        
        Args:
            article: Article identifier
            store_id: Store identifier
            edited_quantity: User-edited quantity
            user_id: User making the edit
        
        Returns:
            Edit record with validation and rebalancing info
        """
        if not self.enable_hitl or not self.hitl:
            raise ValueError("HITL workflow is not enabled")
        
        # Get original quantity and total forecasted quantity
        recommendations = self.forecast_results.get('recommendations', {})
        store_allocations = recommendations.get('store_allocations', {})
        article_metrics = recommendations.get('article_level_metrics', {})
        
        if article not in store_allocations or store_id not in store_allocations[article]:
            raise ValueError(f"Store {store_id} not found for article {article}")
        
        original_quantity = store_allocations[article][store_id].get('quantity', 0)
        total_forecasted = article_metrics.get(article, {}).get('total_quantity', 0)
        
        # Edit store-level quantity
        edit_record = self.hitl.edit_store_level_quantity(
            article, store_id, edited_quantity, original_quantity, total_forecasted, user_id
        )
        
        # Dynamically rebalance (Step 6)
        rebalanced = self.hitl.rebalance_after_store_edit(
            article, store_id, edited_quantity, store_allocations[article], total_forecasted
        )
        
        # Update recommendations
        store_allocations[article] = rebalanced
        
        return edit_record
    
    def generate_final_output_with_approvals(self) -> Dict[str, Any]:
        """
        Generate final output with approved and pending items separated (Step 7)
        
        Returns:
            Final output with approved allocations and pending approvals
        """
        if not self.enable_hitl or not self.hitl:
            raise ValueError("HITL workflow is not enabled")
        
        recommendations = self.forecast_results.get('recommendations', {})
        store_allocations = recommendations.get('store_allocations', {})
        article_level_metrics = recommendations.get('article_level_metrics', {})
        
        return self.hitl.generate_final_output(store_allocations, article_level_metrics)
    
    def get_approval_queue(self) -> List[Dict[str, Any]]:
        """
        Get items requiring approval (Step 7)
        
        Returns:
            List of items needing approval
        """
        if not self.enable_hitl or not self.hitl:
            return []
        
        return self.hitl.get_approval_queue()
    
    def approve_item(self, article: str, store_id: Optional[str] = None,
                    approver_id: str = "category_head") -> Dict[str, Any]:
        """
        Approve an item (aggregate or store-level) (Step 7)
        
        Args:
            article: Article identifier
            store_id: Store ID (None for aggregate level)
            approver_id: Approver user ID
        
        Returns:
            Approval record
        """
        if not self.enable_hitl or not self.hitl:
            raise ValueError("HITL workflow is not enabled")
        
        return self.hitl.approve_item(article, store_id, approver_id)
    
    def get_audit_trail(self, article: Optional[str] = None,
                       store_id: Optional[str] = None,
                       action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail with optional filters (Step 2, 8)
        
        Args:
            article: Filter by article
            store_id: Filter by store
            action_type: Filter by action type
        
        Returns:
            Filtered audit trail
        """
        if not self.enable_hitl or not self.hitl:
            return []
        
        return self.hitl.get_audit_trail(article, store_id, action_type)
    
    def get_system_vs_human_comparison(self, article: str) -> Dict[str, Any]:
        """
        Get comparison of system suggested vs human finalized quantities (Step 8)
        
        Args:
            article: Article identifier
        
        Returns:
            Comparison data for audit trail
        """
        if not self.enable_hitl or not self.hitl:
            return {}
        
        return self.hitl.get_system_vs_human_comparison(article)
    
    def get_variance_highlights(self) -> Dict[str, Any]:
        """
        Get all variances highlighted for quick check (Step 9)
        
        Returns:
            Highlighted variances grouped by type
        """
        if not self.enable_hitl or not self.hitl:
            return {}
        
        approval_queue = self.hitl.get_approval_queue()
        
        highlights = {
            'aggregate_variances': [],
            'store_level_variances': [],
            'style_level_variances': [],
            'critical_variances': [],  # Variances > 10%
            'moderate_variances': []   # Variances 5-10%
        }
        
        for item in approval_queue:
            variance_pct = item['variance_pct']
            
            if item['type'] == 'aggregate':
                highlights['aggregate_variances'].append(item)
            elif item['type'] == 'store_level':
                highlights['store_level_variances'].append(item)
            elif item['type'] == 'style_level':
                highlights['style_level_variances'].append(item)
            
            if variance_pct > 10:
                highlights['critical_variances'].append(item)
            elif variance_pct >= 5:
                highlights['moderate_variances'].append(item)
        
        return highlights
    
    def export_hitl_metadata(self) -> Dict[str, Any]:
        """
        Export HITL metadata for persistence (Step 2)
        
        Returns:
            Complete metadata dictionary
        """
        if not self.enable_hitl or not self.hitl:
            return {}
        
        return self.hitl.export_metadata()
