"""
Weather Data Fetcher
Integrates with multiple weather APIs to provide comprehensive weather data
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    temperature_celsius: float
    humidity_percent: float
    rainfall_mm: float
    wind_speed_kmh: float
    pressure_hpa: float
    uv_index: Optional[float]
    recorded_at: datetime

class WeatherFetcher:
    """
    Multi-source weather data fetcher with fallback mechanisms
    """
    
    def __init__(self):
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.nasa_power_api_key = os.getenv('NASA_POWER_API_KEY')
        
        # API endpoints
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        self.nasa_power_base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Cache for API responses (simple in-memory cache)
        self._cache = {}
        self._cache_duration = 3600  # 1 hour
    
    async def get_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get current weather data for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            
        Returns:
            Dictionary containing current weather data
        """
        try:
            # Try OpenWeatherMap first
            if self.openweather_api_key:
                weather_data = await self._fetch_openweather_current(latitude, longitude)
                if weather_data:
                    return weather_data
            
            # Fallback to NASA POWER (historical data as current)
            weather_data = await self._fetch_nasa_power_current(latitude, longitude)
            if weather_data:
                return weather_data
            
            # Final fallback - mock data
            return self._generate_mock_weather_data()
            
        except Exception as e:
            logger.error(f"Error fetching current weather: {e}")
            return self._generate_mock_weather_data()
    
    async def get_forecast(self, latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
        """
        Get weather forecast for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            days: Number of days to forecast
            
        Returns:
            Dictionary containing forecast data
        """
        try:
            # Try OpenWeatherMap forecast
            if self.openweather_api_key:
                forecast_data = await self._fetch_openweather_forecast(latitude, longitude, days)
                if forecast_data:
                    return forecast_data
            
            # Fallback to NASA POWER historical patterns
            forecast_data = await self._fetch_nasa_power_forecast(latitude, longitude, days)
            if forecast_data:
                return forecast_data
            
            # Final fallback - mock forecast
            return self._generate_mock_forecast_data(days)
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return self._generate_mock_forecast_data(days)
    
    async def get_historical_weather(self, latitude: float, longitude: float,
                                   start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get historical weather data for a date range
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of daily weather data
        """
        try:
            # NASA POWER is better for historical data
            historical_data = await self._fetch_nasa_power_historical(
                latitude, longitude, start_date, end_date
            )
            
            if historical_data:
                return historical_data
            
            # Fallback to mock historical data
            return self._generate_mock_historical_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching historical weather: {e}")
            return self._generate_mock_historical_data(start_date, end_date)
    
    async def _fetch_openweather_current(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Fetch current weather from OpenWeatherMap API"""
        try:
            cache_key = f"owm_current_{latitude}_{longitude}"
            
            # Check cache
            if self._is_cached(cache_key):
                return self._cache[cache_key]['data']
            
            url = f"{self.openweather_base_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0) * 3.6,  # Convert m/s to km/h
                'rainfall': data.get('rain', {}).get('1h', 0),
                'uv_index': None,  # Not available in current weather API
                'description': data['weather'][0]['description'],
                'source': 'openweathermap',
                'timestamp': datetime.now()
            }
            
            # Cache the result
            self._cache[cache_key] = {
                'data': weather_data,
                'timestamp': datetime.now()
            }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching OpenWeatherMap current data: {e}")
            return None
    
    async def _fetch_openweather_forecast(self, latitude: float, longitude: float, 
                                        days: int) -> Optional[Dict[str, Any]]:
        """Fetch weather forecast from OpenWeatherMap API"""
        try:
            cache_key = f"owm_forecast_{latitude}_{longitude}_{days}"
            
            # Check cache
            if self._is_cached(cache_key):
                return self._cache[cache_key]['data']
            
            url = f"{self.openweather_base_url}/forecast"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.openweather_api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse and aggregate daily data
            daily_forecasts = []
            current_date = None
            daily_data = {}
            
            for item in data['list']:
                forecast_date = datetime.fromtimestamp(item['dt']).date()
                
                if current_date != forecast_date:
                    if current_date is not None:
                        daily_forecasts.append(self._aggregate_daily_forecast(daily_data))
                    
                    current_date = forecast_date
                    daily_data = {
                        'date': forecast_date,
                        'temperatures': [],
                        'humidity': [],
                        'rainfall': [],
                        'wind_speed': [],
                        'pressure': []
                    }
                
                # Collect data for aggregation
                daily_data['temperatures'].append(item['main']['temp'])
                daily_data['humidity'].append(item['main']['humidity'])
                daily_data['rainfall'].append(item.get('rain', {}).get('3h', 0))
                daily_data['wind_speed'].append(item['wind'].get('speed', 0) * 3.6)
                daily_data['pressure'].append(item['main']['pressure'])
            
            # Add the last day
            if daily_data:
                daily_forecasts.append(self._aggregate_daily_forecast(daily_data))
            
            forecast_result = {
                'daily': daily_forecasts,
                'source': 'openweathermap',
                'generated_at': datetime.now()
            }
            
            # Cache the result
            self._cache[cache_key] = {
                'data': forecast_result,
                'timestamp': datetime.now()
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error fetching OpenWeatherMap forecast: {e}")
            return None
    
    def _aggregate_daily_forecast(self, daily_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate 3-hourly forecast data into daily summary"""
        return {
            'date': daily_data['date'].isoformat(),
            'temperature_min': min(daily_data['temperatures']),
            'temperature_max': max(daily_data['temperatures']),
            'temperature_avg': sum(daily_data['temperatures']) / len(daily_data['temperatures']),
            'humidity_avg': sum(daily_data['humidity']) / len(daily_data['humidity']),
            'precipitation': sum(daily_data['rainfall']),
            'wind_speed_avg': sum(daily_data['wind_speed']) / len(daily_data['wind_speed']),
            'pressure_avg': sum(daily_data['pressure']) / len(daily_data['pressure'])
        }
    
    async def _fetch_nasa_power_current(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Fetch current weather from NASA POWER API (using recent historical data)"""
        try:
            # NASA POWER doesn't provide real-time data, so we use yesterday's data
            yesterday = datetime.now() - timedelta(days=1)
            
            url = self.nasa_power_base_url
            params = {
                'parameters': 'T2M,RH2M,PRECTOTCORR,WS2M,PS',
                'community': 'AG',
                'longitude': longitude,
                'latitude': latitude,
                'start': yesterday.strftime('%Y%m%d'),
                'end': yesterday.strftime('%Y%m%d'),
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'properties' in data and 'parameter' in data['properties']:
                params_data = data['properties']['parameter']
                date_key = yesterday.strftime('%Y%m%d')
                
                weather_data = {
                    'temperature': params_data.get('T2M', {}).get(date_key, 20),
                    'humidity': params_data.get('RH2M', {}).get(date_key, 60),
                    'rainfall': params_data.get('PRECTOTCORR', {}).get(date_key, 0),
                    'wind_speed': params_data.get('WS2M', {}).get(date_key, 5),
                    'pressure': params_data.get('PS', {}).get(date_key, 1013),
                    'uv_index': None,
                    'description': 'NASA POWER data',
                    'source': 'nasa_power',
                    'timestamp': datetime.now()
                }
                
                return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching NASA POWER current data: {e}")
        
        return None
    
    async def _fetch_nasa_power_forecast(self, latitude: float, longitude: float, 
                                       days: int) -> Optional[Dict[str, Any]]:
        """Generate forecast based on NASA POWER historical patterns"""
        try:
            # Get historical data for the same period in previous years
            current_date = datetime.now()
            historical_forecasts = []
            
            for year_offset in [1, 2, 3]:  # Use last 3 years
                start_date = current_date.replace(year=current_date.year - year_offset)
                end_date = start_date + timedelta(days=days)
                
                historical_data = await self._fetch_nasa_power_historical(
                    latitude, longitude, start_date, end_date
                )
                
                if historical_data:
                    historical_forecasts.extend(historical_data)
            
            if historical_forecasts:
                # Create forecast based on historical averages
                daily_forecasts = []
                
                for day in range(days):
                    forecast_date = current_date + timedelta(days=day)
                    
                    # Find historical data for the same day of year
                    day_of_year = forecast_date.timetuple().tm_yday
                    relevant_data = [
                        d for d in historical_forecasts 
                        if abs(datetime.fromisoformat(d['date']).timetuple().tm_yday - day_of_year) <= 7
                    ]
                    
                    if relevant_data:
                        daily_forecasts.append({
                            'date': forecast_date.date().isoformat(),
                            'temperature_avg': sum(d['temperature'] for d in relevant_data) / len(relevant_data),
                            'humidity_avg': sum(d['humidity'] for d in relevant_data) / len(relevant_data),
                            'precipitation': sum(d['rainfall'] for d in relevant_data) / len(relevant_data),
                            'wind_speed_avg': sum(d['wind_speed'] for d in relevant_data) / len(relevant_data)
                        })
                
                return {
                    'daily': daily_forecasts,
                    'source': 'nasa_power_historical',
                    'generated_at': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Error generating NASA POWER forecast: {e}")
        
        return None
    
    async def _fetch_nasa_power_historical(self, latitude: float, longitude: float,
                                         start_date: datetime, end_date: datetime) -> Optional[List[Dict[str, Any]]]:
        """Fetch historical weather data from NASA POWER API"""
        try:
            url = self.nasa_power_base_url
            params = {
                'parameters': 'T2M,RH2M,PRECTOTCORR,WS2M,PS',
                'community': 'AG',
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.strftime('%Y%m%d'),
                'end': end_date.strftime('%Y%m%d'),
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            if 'properties' in data and 'parameter' in data['properties']:
                params_data = data['properties']['parameter']
                
                historical_data = []
                current_date = start_date
                
                while current_date <= end_date:
                    date_key = current_date.strftime('%Y%m%d')
                    
                    weather_record = {
                        'date': current_date.date().isoformat(),
                        'temperature': params_data.get('T2M', {}).get(date_key, 20),
                        'humidity': params_data.get('RH2M', {}).get(date_key, 60),
                        'rainfall': params_data.get('PRECTOTCORR', {}).get(date_key, 0),
                        'wind_speed': params_data.get('WS2M', {}).get(date_key, 5),
                        'pressure': params_data.get('PS', {}).get(date_key, 1013),
                        'source': 'nasa_power'
                    }
                    
                    historical_data.append(weather_record)
                    current_date += timedelta(days=1)
                
                return historical_data
        
        except Exception as e:
            logger.error(f"Error fetching NASA POWER historical data: {e}")
        
        return None
    
    def _generate_mock_weather_data(self) -> Dict[str, Any]:
        """Generate mock weather data as fallback"""
        import random
        
        return {
            'temperature': random.uniform(15, 35),
            'humidity': random.uniform(40, 80),
            'rainfall': random.uniform(0, 10),
            'wind_speed': random.uniform(5, 20),
            'pressure': random.uniform(1000, 1020),
            'uv_index': random.uniform(3, 8),
            'description': 'Mock weather data',
            'source': 'mock',
            'timestamp': datetime.now()
        }
    
    def _generate_mock_forecast_data(self, days: int) -> Dict[str, Any]:
        """Generate mock forecast data as fallback"""
        import random
        
        daily_forecasts = []
        
        for day in range(days):
            forecast_date = datetime.now() + timedelta(days=day)
            
            daily_forecasts.append({
                'date': forecast_date.date().isoformat(),
                'temperature_min': random.uniform(10, 20),
                'temperature_max': random.uniform(25, 35),
                'temperature_avg': random.uniform(18, 28),
                'humidity_avg': random.uniform(50, 80),
                'precipitation': random.uniform(0, 15),
                'wind_speed_avg': random.uniform(5, 15)
            })
        
        return {
            'daily': daily_forecasts,
            'source': 'mock',
            'generated_at': datetime.now()
        }
    
    def _generate_mock_historical_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Generate mock historical data as fallback"""
        import random
        
        historical_data = []
        current_date = start_date
        
        while current_date <= end_date:
            weather_record = {
                'date': current_date.date().isoformat(),
                'temperature': random.uniform(15, 30),
                'humidity': random.uniform(50, 80),
                'rainfall': random.uniform(0, 20),
                'wind_speed': random.uniform(5, 15),
                'pressure': random.uniform(1005, 1015),
                'source': 'mock'
            }
            
            historical_data.append(weather_record)
            current_date += timedelta(days=1)
        
        return historical_data
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self._cache:
            return False
        
        cached_time = self._cache[cache_key]['timestamp']
        return (datetime.now() - cached_time).seconds < self._cache_duration
    
    def clear_cache(self):
        """Clear the weather data cache"""
        self._cache.clear()
        logger.info("Weather data cache cleared")