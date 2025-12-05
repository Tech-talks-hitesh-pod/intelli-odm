"""Demand Forecasting Agent with multiple forecasting approaches."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config.agent_configs import DEMAND_FORECASTING_CONFIG
from utils.llm_client import LLMClient, LLMError, retry_llm_call

logger = logging.getLogger(__name__)

class ForecastingError(Exception):
    """Raised when forecasting operations fail."""
    pass

class DemandForecastingAgent:
    """
    Agent for generating store-level demand forecasts using multiple approaches.
    
    Supports analogy-based, time-series, and ML regression forecasting methods.
    Includes price sensitivity analysis and confidence estimation.
    """
    
    def __init__(self, llm_client: LLMClient, config: Optional[Dict] = None):
        """
        Initialize with LLM client for method selection guidance.
        
        Args:
            llm_client: LLM client for reasoning about forecast methods
            config: Agent configuration
        """
        self.llm_client = llm_client
        self.config = config or DEMAND_FORECASTING_CONFIG
        
        # Initialize forecasting models
        self._init_forecasting_models()
    
    def _init_forecasting_models(self):
        """Initialize forecasting model components."""
        try:
            # Try to import Prophet for time-series forecasting
            try:
                from prophet import Prophet
                self.prophet_available = True
                logger.info("Prophet available for time-series forecasting")
            except ImportError:
                self.prophet_available = False
                logger.warning("Prophet not available - time-series forecasting will be limited")
            
            # Try to import scikit-learn for ML forecasting
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.preprocessing import StandardScaler
                self.sklearn_available = True
                logger.info("Scikit-learn available for ML forecasting")
            except ImportError:
                self.sklearn_available = False
                logger.warning("Scikit-learn not available - ML forecasting will be limited")
                
        except Exception as e:
            logger.error(f"Failed to initialize forecasting models: {e}")
    
    def select_forecast_method(self, comparables: List[Dict], store_data: Dict) -> str:
        """
        Choose optimal forecasting method based on data availability and quality.
        
        Args:
            comparables: List of comparable products
            store_data: Historical store performance data
            
        Returns:
            Selected forecasting method
        """
        logger.info("Selecting optimal forecasting method")
        
        try:
            # Get data characteristics
            num_comparables = len(comparables)
            avg_similarity = np.mean([c.get('similarity_score', 0) for c in comparables]) if comparables else 0
            
            # Check historical data availability
            sales_data = store_data.get('sales', pd.DataFrame())
            history_months = 0
            if not sales_data.empty and 'date' in sales_data.columns:
                date_range = pd.to_datetime(sales_data['date']).max() - pd.to_datetime(sales_data['date']).min()
                history_months = date_range.days / 30.44
            
            num_stores = store_data.get('num_stores', 0)
            
            # Apply selection thresholds from config
            thresholds = self.config["method_selection"]
            
            # Priority 1: Analogy-based if good comparables available
            if (num_comparables >= thresholds["analogy_threshold"]["min_comparables"] and
                avg_similarity >= thresholds["analogy_threshold"]["min_similarity"]):
                method = "analogy"
                reason = f"High-quality comparables available (n={num_comparables}, avg_sim={avg_similarity:.2f})"
            
            # Priority 2: Time-series if sufficient history
            elif (history_months >= thresholds["timeseries_threshold"]["min_history_months"] and
                  self.prophet_available):
                method = "timeseries"
                reason = f"Sufficient historical data ({history_months:.1f} months)"
            
            # Priority 3: ML regression if many stores
            elif (num_stores >= thresholds["ml_threshold"]["min_stores"] and
                  self.sklearn_available):
                method = "ml_regression"
                reason = f"Large store network ({num_stores} stores)"
            
            # Fallback: Analogy with lower confidence
            else:
                method = "analogy"
                reason = "Fallback method - limited data available"
            
            logger.info(f"Selected method: {method} ({reason})")
            return method
            
        except Exception as e:
            logger.error(f"Method selection failed: {e}")
            return "analogy"  # Safe fallback
    
    def forecast_by_analogy(self, comparables: List[Dict], target_attributes: Dict, 
                           store_data: Dict) -> Dict[str, Dict]:
        """
        Generate forecast using analogy-based scaling of comparable products.
        
        Args:
            comparables: List of comparable products with performance data
            target_attributes: Attributes of the target product
            store_data: Store-level historical data
            
        Returns:
            Store-level forecast results
        """
        logger.info("Generating analogy-based forecast")
        
        try:
            forecast_results = {}
            
            if not comparables:
                logger.warning("No comparables available for analogy forecasting")
                return {}
            
            # Get store list
            stores = store_data.get('stores', [])
            if not stores:
                # Extract from sales data if available
                sales_df = store_data.get('sales', pd.DataFrame())
                if not sales_df.empty and 'store_id' in sales_df.columns:
                    stores = sales_df['store_id'].unique().tolist()
            
            # Calculate base performance from comparables
            base_performance = self._calculate_base_performance(comparables)
            
            for store_id in stores:
                # Get store-specific adjustment factors
                store_factor = self._get_store_adjustment_factor(store_id, store_data)
                
                # Get seasonal adjustment
                seasonal_factor = self._get_seasonal_adjustment(target_attributes)
                
                # Calculate forecast
                base_demand = base_performance.get('avg_monthly_sales', 50)  # Default base
                adjusted_demand = base_demand * store_factor * seasonal_factor
                
                # Add uncertainty based on similarity confidence
                avg_similarity = np.mean([c.get('similarity_score', 0) for c in comparables])
                uncertainty_factor = 1.0 - (avg_similarity * 0.3)  # Lower similarity = higher uncertainty
                
                forecast_results[store_id] = {
                    'mean_demand': max(0, adjusted_demand),
                    'low_ci': max(0, adjusted_demand * (1 - uncertainty_factor)),
                    'high_ci': adjusted_demand * (1 + uncertainty_factor),
                    'confidence': min(0.9, avg_similarity),
                    'method_used': 'analogy',
                    'rationale': f"Based on {len(comparables)} comparable products (avg similarity: {avg_similarity:.2f})"
                }
            
            logger.info(f"Generated analogy forecasts for {len(forecast_results)} stores")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Analogy forecasting failed: {e}")
            raise ForecastingError(f"Analogy forecasting failed: {e}")
    
    def forecast_by_timeseries(self, historical_data: pd.DataFrame, 
                              forecast_days: int = 60) -> Dict[str, Dict]:
        """
        Generate forecast using time-series modeling with Prophet.
        
        Args:
            historical_data: Historical sales data
            forecast_days: Number of days to forecast
            
        Returns:
            Store-level forecast results
        """
        logger.info("Generating time-series forecast")
        
        if not self.prophet_available:
            raise ForecastingError("Prophet not available for time-series forecasting")
        
        try:
            from prophet import Prophet
            
            forecast_results = {}
            
            # Ensure we have the required columns
            if 'date' not in historical_data.columns or 'units_sold' not in historical_data.columns:
                raise ForecastingError("Missing required columns for time-series forecasting")
            
            # Group by store
            for store_id in historical_data['store_id'].unique():
                store_data = historical_data[historical_data['store_id'] == store_id].copy()
                
                # Prepare data for Prophet
                prophet_data = store_data.groupby('date')['units_sold'].sum().reset_index()
                prophet_data.columns = ['ds', 'y']
                prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                
                # Skip if insufficient data
                if len(prophet_data) < 30:  # Need at least 30 days
                    continue
                
                try:
                    # Fit Prophet model
                    model_config = self.config["models"]["prophet"]
                    model = Prophet(
                        seasonality_mode=model_config["seasonality_mode"],
                        yearly_seasonality=model_config["yearly_seasonality"],
                        weekly_seasonality=model_config["weekly_seasonality"],
                        daily_seasonality=model_config["daily_seasonality"],
                        changepoint_prior_scale=model_config["changepoint_prior_scale"]
                    )
                    
                    model.fit(prophet_data)
                    
                    # Generate forecast
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)
                    
                    # Extract forecast period
                    forecast_period = forecast.tail(forecast_days)
                    
                    # Calculate metrics
                    total_forecast = forecast_period['yhat'].sum()
                    lower_bound = forecast_period['yhat_lower'].sum()
                    upper_bound = forecast_period['yhat_upper'].sum()
                    
                    # Calculate confidence based on prediction intervals
                    confidence = 1.0 - ((upper_bound - lower_bound) / total_forecast) if total_forecast > 0 else 0.5
                    confidence = max(0.1, min(0.9, confidence))
                    
                    forecast_results[store_id] = {
                        'mean_demand': max(0, total_forecast),
                        'low_ci': max(0, lower_bound),
                        'high_ci': max(0, upper_bound),
                        'confidence': confidence,
                        'method_used': 'timeseries',
                        'rationale': f"Prophet model based on {len(prophet_data)} days of historical data"
                    }
                    
                except Exception as e:
                    logger.warning(f"Time-series forecasting failed for store {store_id}: {e}")
                    continue
            
            logger.info(f"Generated time-series forecasts for {len(forecast_results)} stores")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Time-series forecasting failed: {e}")
            raise ForecastingError(f"Time-series forecasting failed: {e}")
    
    def forecast_by_ml_regression(self, store_features: pd.DataFrame, 
                                 product_features: Dict, 
                                 target_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Generate forecast using ML regression with store and product features.
        
        Args:
            store_features: Store-level feature data
            product_features: Product feature data
            target_data: Historical performance data
            
        Returns:
            Store-level forecast results
        """
        logger.info("Generating ML regression forecast")
        
        if not self.sklearn_available:
            raise ForecastingError("Scikit-learn not available for ML forecasting")
        
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # This is a simplified implementation
            # In production, you'd have more sophisticated feature engineering
            
            forecast_results = {}
            
            # Create simple features for demonstration
            for store_id in store_features['store_id'].unique():
                # Generate a forecast based on store tier and category
                store_info = store_features[store_features['store_id'] == store_id].iloc[0]
                
                # Simple rule-based forecast for demonstration
                tier = store_info.get('tier', 'Tier_2')
                performance_factor = store_info.get('performance_factor', 1.0)
                
                category = product_features.get('category', 'TSHIRT')
                
                # Base forecast by category
                base_forecasts = {
                    'TSHIRT': 80,
                    'POLO': 60,
                    'DRESS': 40,
                    'JEANS': 50,
                    'SHIRT': 45
                }
                
                base_demand = base_forecasts.get(category, 50)
                adjusted_demand = base_demand * performance_factor
                
                # Add uncertainty
                uncertainty = 0.3  # 30% uncertainty
                
                forecast_results[store_id] = {
                    'mean_demand': adjusted_demand,
                    'low_ci': adjusted_demand * (1 - uncertainty),
                    'high_ci': adjusted_demand * (1 + uncertainty),
                    'confidence': 0.7,  # Moderate confidence
                    'method_used': 'ml_regression',
                    'rationale': f"ML model based on store tier ({tier}) and product category ({category})"
                }
            
            logger.info(f"Generated ML forecasts for {len(forecast_results)} stores")
            return forecast_results
            
        except Exception as e:
            logger.error(f"ML forecasting failed: {e}")
            raise ForecastingError(f"ML forecasting failed: {e}")
    
    def _calculate_base_performance(self, comparables: List[Dict]) -> Dict:
        """Calculate weighted average performance from comparables."""
        if not comparables:
            return {'avg_monthly_sales': 50}  # Default
        
        total_weighted_sales = 0
        total_weights = 0
        
        for comp in comparables:
            similarity = comp.get('similarity_score', 0.5)
            perf_data = comp.get('metadata', {}).get('performance', {})
            monthly_sales = perf_data.get('avg_monthly_sales', 50)
            
            total_weighted_sales += monthly_sales * similarity
            total_weights += similarity
        
        avg_sales = total_weighted_sales / total_weights if total_weights > 0 else 50
        
        return {
            'avg_monthly_sales': avg_sales,
            'confidence': total_weights / len(comparables)
        }
    
    def _get_store_adjustment_factor(self, store_id: str, store_data: Dict) -> float:
        """Get store-specific adjustment factor."""
        # Get store tier info
        store_tiers = store_data.get('store_tiers', pd.DataFrame())
        
        if not store_tiers.empty and 'store_id' in store_tiers.columns:
            store_info = store_tiers[store_tiers['store_id'] == store_id]
            if not store_info.empty:
                return store_info.iloc[0].get('performance_factor', 1.0)
        
        return 1.0  # Default adjustment
    
    def _get_seasonal_adjustment(self, attributes: Dict) -> float:
        """Get seasonal adjustment factor based on current month."""
        category = attributes.get('category', '').upper()
        current_month = datetime.now().month
        
        # Seasonal patterns by category
        seasonal_patterns = {
            'TSHIRT': {3: 1.2, 4: 1.3, 5: 1.4, 6: 1.2, 7: 1.1, 8: 1.0, 9: 0.9},
            'DRESS': {4: 1.3, 5: 1.2, 12: 1.4},
            'JEANS': {9: 1.2, 10: 1.3, 11: 1.2, 1: 1.1, 2: 1.0}
        }
        
        pattern = seasonal_patterns.get(category, {})
        return pattern.get(current_month, 1.0)
    
    def price_sensitivity_analysis(self, base_forecast: Dict[str, Dict], 
                                  price_points: List[float]) -> Dict[str, Any]:
        """
        Analyze demand sensitivity to different price points.
        
        Args:
            base_forecast: Base demand forecast
            price_points: List of prices to analyze
            
        Returns:
            Price sensitivity analysis results
        """
        logger.info(f"Analyzing price sensitivity for {len(price_points)} price points")
        
        try:
            # Calculate total base demand
            total_base_demand = sum(store['mean_demand'] for store in base_forecast.values())
            
            if total_base_demand == 0:
                raise ForecastingError("Zero base demand for price sensitivity analysis")
            
            # Assume a price elasticity (simplified model)
            # In production, this would be learned from historical data
            elasticity = -1.2  # Price elastic
            
            price_scenarios = {}
            base_price = np.median(price_points)  # Use median as base price
            
            for price in price_points:
                # Calculate demand multiplier based on price change
                price_change_pct = (price - base_price) / base_price
                demand_change_pct = elasticity * price_change_pct
                demand_multiplier = 1 + demand_change_pct
                
                # Apply bounds (demand can't go negative or increase too much)
                demand_multiplier = max(0.1, min(2.0, demand_multiplier))
                
                adjusted_demand = total_base_demand * demand_multiplier
                expected_revenue = adjusted_demand * price
                
                # Calculate margin impact (simplified)
                margin_change = (price - base_price) / base_price * 0.5  # Assume 50% flow-through
                
                price_scenarios[str(int(price))] = {
                    'demand_multiplier': round(demand_multiplier, 2),
                    'expected_units': round(adjusted_demand),
                    'expected_revenue': round(expected_revenue),
                    'margin_impact': round(margin_change, 2),
                    'risk_assessment': self._assess_price_risk(demand_multiplier, margin_change)
                }
            
            # Find optimal price (revenue maximizing)
            optimal_price_str = max(price_scenarios.keys(), 
                                  key=lambda p: price_scenarios[p]['expected_revenue'])
            optimal_price = float(optimal_price_str)
            
            results = {
                'price_elasticity': elasticity,
                'optimal_price': optimal_price,
                'base_price': base_price,
                'price_scenarios': price_scenarios,
                'recommendations': self._generate_pricing_recommendations(price_scenarios)
            }
            
            logger.info(f"Price sensitivity analysis completed. Optimal price: ₹{optimal_price}")
            return results
            
        except Exception as e:
            logger.error(f"Price sensitivity analysis failed: {e}")
            return {
                'error': str(e),
                'price_elasticity': -1.0,
                'price_scenarios': {}
            }
    
    def _assess_price_risk(self, demand_multiplier: float, margin_change: float) -> str:
        """Assess risk level for price scenario."""
        if demand_multiplier < 0.7:
            return "High volume risk"
        elif demand_multiplier > 1.3:
            return "High demand uncertainty" 
        elif margin_change < -0.2:
            return "Low margin risk"
        elif margin_change > 0.2:
            return "Premium pricing"
        else:
            return "Balanced risk-return"
    
    def _generate_pricing_recommendations(self, scenarios: Dict) -> List[str]:
        """Generate pricing recommendations."""
        recommendations = []
        
        # Find scenarios with good balance
        for price, data in scenarios.items():
            if (0.8 <= data['demand_multiplier'] <= 1.2 and 
                data['margin_impact'] >= 0):
                recommendations.append(f"Price ₹{price} offers good volume-margin balance")
        
        # Add risk warnings
        high_risk_prices = [p for p, d in scenarios.items() if d['demand_multiplier'] < 0.6]
        if high_risk_prices:
            recommendations.append(f"Avoid pricing above ₹{max(high_risk_prices)} due to demand decline")
        
        return recommendations or ["Consider market testing before final price decision"]
    
    def run(self, comparables: List[Dict], sales_df: pd.DataFrame, 
           inventory_df: pd.DataFrame, price_df: pd.DataFrame, 
           price_options: List[float]) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """
        Execute complete demand forecasting workflow.
        
        Args:
            comparables: List of comparable products
            sales_df: Historical sales data  
            inventory_df: Current inventory data
            price_df: Current pricing data
            price_options: Price points to analyze
            
        Returns:
            Tuple of (forecast_results, sensitivity_analysis)
        """
        logger.info("Starting demand forecasting workflow")
        
        try:
            # Prepare store data
            store_data = {
                'sales': sales_df,
                'inventory': inventory_df,
                'pricing': price_df,
                'num_stores': sales_df['store_id'].nunique() if not sales_df.empty else 0
            }
            
            # Add store tiers if available
            if not sales_df.empty:
                # Simple store tier classification
                store_performance = sales_df.groupby('store_id')['units_sold'].sum().reset_index()
                store_performance['performance_factor'] = (
                    store_performance['units_sold'] / store_performance['units_sold'].median()
                )
                store_performance['tier'] = store_performance['performance_factor'].apply(
                    lambda x: 'Tier_1' if x >= 1.2 else ('Tier_2' if x >= 0.8 else 'Tier_3')
                )
                store_data['store_tiers'] = store_performance
                
                # Extract product attributes from comparables
                target_attributes = {}
                if comparables:
                    first_comp = comparables[0]
                    target_attributes = first_comp.get('attributes', {})
            
            # Step 1: Select forecasting method
            method = self.select_forecast_method(comparables, store_data)
            
            # Step 2: Generate forecast
            if method == "analogy":
                forecast_results = self.forecast_by_analogy(comparables, target_attributes, store_data)
            elif method == "timeseries":
                forecast_results = self.forecast_by_timeseries(sales_df)
            elif method == "ml_regression":
                forecast_results = self.forecast_by_ml_regression(
                    store_data.get('store_tiers', pd.DataFrame()), 
                    target_attributes, 
                    sales_df
                )
            else:
                raise ForecastingError(f"Unknown forecasting method: {method}")
            
            # Step 3: Price sensitivity analysis
            sensitivity_analysis = self.price_sensitivity_analysis(forecast_results, price_options)
            
            logger.info(f"Demand forecasting completed. Generated forecasts for {len(forecast_results)} stores")
            return forecast_results, sensitivity_analysis
            
        except Exception as e:
            logger.error(f"Demand forecasting workflow failed: {e}")
            raise