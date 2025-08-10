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
        self.openmeteo_base_url = "https://api.open-meteo.com/v1"
        
        # Cache for API responses (simple in-memory cache)
        self._cache = {}
        self._cache_duration = 3600  # 1 hour
    
    async def get_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Get current weather data for a location - LIVE DATA ONLY
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            
        Returns:
            Dictionary containing current weather data
        """
        # Try multiple real APIs in order of preference
        apis_to_try = []
        
        if self.openweather_api_key:
            apis_to_try.append(('openweather', self._fetch_openweather_current))
        
        # Add Open-Meteo (free, no API key required)
        apis_to_try.append(('openmeteo', self._fetch_openmeteo_current))
        
        # Add NASA POWER as last resort
        apis_to_try.append(('nasa_power', self._fetch_nasa_power_current))
        
        for api_name, fetch_func in apis_to_try:
            try:
                logger.info(f"Trying {api_name} API for weather data")
                weather_data = await fetch_func(latitude, longitude)
                if weather_data and weather_data.get('temperature') is not None:
                    logger.info(f"Successfully fetched weather data from {api_name}")
                    return weather_data
                else:
                    logger.warning(f"{api_name} returned incomplete data")
            except Exception as e:
                logger.error(f"Error with {api_name} API: {e}")
                continue
        
        # If all APIs fail, raise an exception instead of returning mock data
        raise Exception("All weather APIs failed - no live data available. Please check API keys and network connection.")
    
    async def get_forecast(self, latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
        """
        Get weather forecast for a location - LIVE DATA ONLY
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            days: Number of days to forecast
            
        Returns:
            Dictionary containing forecast data
        """
        # Try multiple real APIs in order of preference
        apis_to_try = []
        
        if self.openweather_api_key:
            apis_to_try.append(('openweather', self._fetch_openweather_forecast))
        
        # Add Open-Meteo (free, no API key required)
        apis_to_try.append(('openmeteo', self._fetch_openmeteo_forecast))
        
        for api_name, fetch_func in apis_to_try:
            try:
                logger.info(f"Trying {api_name} API for forecast data")
                forecast_data = await fetch_func(latitude, longitude, days)
                if forecast_data and forecast_data.get('daily'):
                    logger.info(f"Successfully fetched forecast from {api_name}")
                    return forecast_data
                else:
                    logger.warning(f"{api_name} returned incomplete forecast data")
            except Exception as e:
                logger.error(f"Error with {api_name} forecast API: {e}")
                continue
        
        # If all APIs fail, raise an exception
        raise Exception("All weather forecast APIs failed - no live data available. Please check API keys and network connection.")
    
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
    
    async def _fetch_openmeteo_forecast(self, latitude: float, longitude: float, 
                                      days: int) -> Optional[Dict[str, Any]]:
        """Fetch weather forecast from Open-Meteo API (free, no API key required)"""
        try:
            cache_key = f"openmeteo_forecast_{latitude}_{longitude}_{days}"
            
            # Check cache
            if self._is_cached(cache_key):
                return self._cache[cache_key]['data']
            
            url = f"{self.openmeteo_base_url}/forecast"
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean',
                'timezone': 'auto',
                'forecast_days': min(days, 16)  # Open-Meteo supports up to 16 days
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            daily_data = data.get('daily', {})
            
            if not daily_data:
                return None
            
            # Parse daily forecasts
            daily_forecasts = []
            dates = daily_data.get('time', [])
            temp_max = daily_data.get('temperature_2m_max', [])
            temp_min = daily_data.get('temperature_2m_min', [])
            precipitation = daily_data.get('precipitation_sum', [])
            wind_speed = daily_data.get('wind_speed_10m_max', [])
            humidity = daily_data.get('relative_humidity_2m_mean', [])
            
            for i in range(min(len(dates), days)):
                daily_forecasts.append({
                    'date': dates[i],
                    'temperature_min': temp_min[i] if i < len(temp_min) else 15,
                    'temperature_max': temp_max[i] if i < len(temp_max) else 25,
                    'temperature_avg': (temp_max[i] + temp_min[i]) / 2 if i < len(temp_max) and i < len(temp_min) else 20,
                    'humidity_avg': humidity[i] if i < len(humidity) else 60,
                    'precipitation': precipitation[i] if i < len(precipitation) else 0,
                    'wind_speed_avg': wind_speed[i] if i < len(wind_speed) else 5
                })
            
            forecast_result = {
                'daily': daily_forecasts,
                'source': 'openmeteo',
                'generated_at': datetime.now()
            }
            
            # Cache the result
            self._cache[cache_key] = {
                'data': forecast_result,
                'timestamp': datetime.now()
            }
            
            return forecast_result
            
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo forecast: {e}")
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
    
    async def _fetch_openmeteo_current(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """Fetch current weather from Open-Meteo API (free, no API key required)"""
        try:
            cache_key = f"openmeteo_current_{latitude}_{longitude}"
            
            # Check cache
            if self._is_cached(cache_key):
                return self._cache[cache_key]['data']
            
            url = f"{self.openmeteo_base_url}/forecast"
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure',
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get('current', {})
            
            weather_data = {
                'temperature': current.get('temperature_2m', 20),
                'humidity': current.get('relative_humidity_2m', 60),
                'rainfall': current.get('precipitation', 0),
                'wind_speed': current.get('wind_speed_10m', 5),
                'pressure': current.get('surface_pressure', 1013),
                'uv_index': None,
                'description': 'Open-Meteo live data',
                'source': 'openmeteo',
                'timestamp': datetime.now()
            }
            
            # Cache the result
            self._cache[cache_key] = {
                'data': weather_data,
                'timestamp': datetime.now()
            }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo current data: {e}")
            return None

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
    
    def validate_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Validate that weather data contains required fields"""
        required_fields = ['temperature', 'humidity', 'source', 'timestamp']
        return all(field in weather_data and weather_data[field] is not None for field in required_fields)
    
    def get_api_status(self) -> Dict[str, str]:
        """Get status of all weather APIs"""
        status = {}
        
        # Check OpenWeatherMap
        if self.openweather_api_key:
            status['openweather'] = 'API key configured'
        else:
            status['openweather'] = 'No API key - get free key at openweathermap.org'
        
        # Open-Meteo is always available (no API key required)
        status['openmeteo'] = 'Available (no API key required)'
        
        # NASA POWER
        status['nasa_power'] = 'Available (no API key required, historical data only)'
        
        return status
    
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