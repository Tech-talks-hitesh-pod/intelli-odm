# agents/demand_forecasting_agent.py

class DemandForecastingAgent:
    def __init__(self, llama_client):
        self.llama = llama_client

    def select_model(self, comparables):
        # Use LLM to guide model selection
        # For now, return a simple model identifier
        return "time_series_model"

    def forecast(self, sales_data, inventory_data, price_data):
        # Create probabilistic forecasts
        # For now, return a simple forecast structure
        return {
            "forecasted_demand": 1000,
            "confidence_interval": (800, 1200),
            "time_period": "next_month"
        }

    def sensitivity_analysis(self, forecast, price_options):
        # Predict impact of different prices
        try:
            prompt = f"""Analyze demand sensitivity for these price options: {price_options}
            Base forecast: {forecast.get('forecasted_demand', 'N/A')}
            
            Provide a brief analysis of how demand might change with different prices."""
            
            response = self.llama.generate(
                prompt=prompt,
                system="You are a demand forecasting expert specializing in price sensitivity analysis."
            )
            return {
                "price_sensitivity": response,
                "price_options": price_options,
                "base_forecast": forecast
            }
        except Exception as e:
            print(f"Error in sensitivity_analysis: {e}")
            return {
                "price_sensitivity": "Analysis unavailable",
                "price_options": price_options,
                "base_forecast": forecast
            }

    def run(self, comparables, sales_data, inventory_data, price_data, price_options):
        model = self.select_model(comparables)
        forecast = self.forecast(sales_data, inventory_data, price_data)
        analysis = self.sensitivity_analysis(forecast, price_options)
        return forecast, analysis