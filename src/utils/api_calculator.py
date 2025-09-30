import numpy as np

class AQICalculator:
    """Calculate AQI from pollutant concentrations"""
    
    # AQI breakpoints (concentration ranges)
    PM25_BREAKPOINTS = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500, 301, 500)
    ]
    
    @staticmethod
    def calculate_aqi(pollutant, concentration):
        """Calculate AQI for a given pollutant"""
        if pollutant == 'PM2.5':
            breakpoints = AQICalculator.PM25_BREAKPOINTS
        else:
            # Add other pollutants as needed
            return None
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * \
                      (concentration - bp_lo) + aqi_lo
                return round(aqi)
        
        return 500  # Hazardous
    
    @staticmethod
    def get_aqi_category(aqi):
        """Get AQI category from score"""
        if aqi <= 50:
            return "Good", "green"
        elif aqi <= 100:
            return "Moderate", "yellow"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "orange"
        elif aqi <= 200:
            return "Unhealthy", "red"
        elif aqi <= 300:
            return "Very Unhealthy", "purple"
        else:
            return "Hazardous", "maroon"