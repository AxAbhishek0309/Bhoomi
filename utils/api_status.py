"""
API Status Checker
Monitors the health and availability of all external APIs
"""

import os
import requests
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass

from utils.weather_fetcher import WeatherFetcher
from utils.satellite_ndvi import NDVIFetcher
from agents.market_forecaster import MarketForecastAgent

logger = logging.getLogger(__name__)

@dataclass
class APIStatus:
    name: str
    status: str  # 'online', 'offline', 'error', 'no_key'
    response_time_ms: float
    last_checked: datetime
    error_message: str = None

class APIStatusChecker:
    """
    Comprehensive API status checker for all external services
    """
    
    def __init__(self):
        self.weather_fetcher = WeatherFetcher()
        self.ndvi_fetcher = NDVIFetcher()
        self.market_agent = MarketForecastAgent({})
    
    async def check_all_apis(self) -> Dict[str, APIStatus]:
        """Check status of all APIs"""
        results = {}
        
        # Check weather APIs
        weather_apis = await self._check_weather_apis()
        results.update(weather_apis)
        
        # Check satellite APIs
        satellite_apis = await self._check_satellite_apis()
        results.update(satellite_apis)
        
        # Check market APIs
        market_apis = await self._check_market_apis()
        results.update(market_apis)
        
        # Check other APIs
        other_apis = await self._check_other_apis()
        results.update(other_apis)
        
        return results
    
    async def _check_weather_apis(self) -> Dict[str, APIStatus]:
        """Check weather API status"""
        results = {}
        
        # OpenWeatherMap
        if os.getenv('OPENWEATHER_API_KEY'):
            results['openweather'] = await self._check_openweather()
        else:
            results['openweather'] = APIStatus(
                name='OpenWeatherMap',
                status='no_key',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message='API key not configured'
            )
        
        # Open-Meteo (free, no key required)
        results['openmeteo'] = await self._check_openmeteo()
        
        # NASA POWER
        results['nasa_power'] = await self._check_nasa_power()
        
        return results
    
    async def _check_satellite_apis(self) -> Dict[str, APIStatus]:
        """Check satellite API status"""
        results = {}
        
        # NASA MODIS (real API test)
        results['nasa_modis'] = await self._check_nasa_modis()
        
        # NASA EarthData
        results['nasa_earthdata'] = await self._check_nasa_earthdata()
        
        # USGS Landsat
        results['usgs_landsat'] = await self._check_usgs_landsat()
        
        return results
    
    async def _check_market_apis(self) -> Dict[str, APIStatus]:
        """Check market data API status"""
        results = {}
        
        # Agmarknet Scraper
        results['agmarknet_scraper'] = await self._check_agmarknet_scraper()
        
        # data.gov.in
        results['data_gov_in'] = await self._check_data_gov_in()
        
        return results
    
    async def _check_other_apis(self) -> Dict[str, APIStatus]:
        """Check other API status"""
        results = {}
        
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            results['openai'] = await self._check_openai()
        else:
            results['openai'] = APIStatus(
                name='OpenAI',
                status='no_key',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message='API key not configured'
            )
        
        return results
    
    async def _check_openweather(self) -> APIStatus:
        """Check OpenWeatherMap API"""
        try:
            start_time = datetime.now()
            
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': 28.6139,  # Delhi coordinates for test
                'lon': 77.2090,
                'appid': os.getenv('OPENWEATHER_API_KEY'),
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'main' in data and 'temp' in data['main']:
                    return APIStatus(
                        name='OpenWeatherMap',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='OpenWeatherMap',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}: {response.text[:100]}"
            )
            
        except Exception as e:
            return APIStatus(
                name='OpenWeatherMap',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_openmeteo(self) -> APIStatus:
        """Check Open-Meteo API"""
        try:
            start_time = datetime.now()
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': 28.6139,
                'longitude': 77.2090,
                'current': 'temperature_2m',
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'current' in data:
                    return APIStatus(
                        name='Open-Meteo',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='Open-Meteo',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='Open-Meteo',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_nasa_power(self) -> APIStatus:
        """Check NASA POWER API"""
        try:
            start_time = datetime.now()
            
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                'parameters': 'T2M',
                'community': 'AG',
                'longitude': 77.2090,
                'latitude': 28.6139,
                'start': '20240101',
                'end': '20240102',
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'properties' in data:
                    return APIStatus(
                        name='NASA POWER',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='NASA POWER',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='NASA POWER',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_agmarknet_scraper(self) -> APIStatus:
        """Check Agmarknet website accessibility for scraping"""
        try:
            start_time = datetime.now()
            
            url = "https://agmarknet.gov.in"
            
            response = requests.get(url, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return APIStatus(
                    name='Agmarknet Scraper',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='Agmarknet Scraper',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='Agmarknet Scraper',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_nasa_modis(self) -> APIStatus:
        """Check NASA MODIS ORNL DAAC API"""
        try:
            start_time = datetime.now()
            
            # Test the MODIS ORNL DAAC API
            url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates"
            params = {
                'latitude': 28.6139,  # Delhi coordinates for test
                'longitude': 77.2090,
                'startDate': '2023-01-01',
                'endDate': '2023-01-31'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'dates' in data:
                    return APIStatus(
                        name='NASA MODIS ORNL',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='NASA MODIS ORNL',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}: Invalid response format"
            )
            
        except Exception as e:
            return APIStatus(
                name='NASA MODIS ORNL',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_nasa_earthdata(self) -> APIStatus:
        """Check NASA EarthData Search API"""
        try:
            start_time = datetime.now()
            
            url = "https://cmr.earthdata.nasa.gov/search/collections.json"
            params = {
                'page_size': 1,
                'keyword': 'MODIS'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    return APIStatus(
                        name='NASA EarthData Search',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='NASA EarthData Search',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='NASA EarthData Search',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_usgs_landsat(self) -> APIStatus:
        """Check USGS Landsat availability"""
        try:
            start_time = datetime.now()
            
            # Test basic USGS connectivity
            url = "https://landsatlook.usgs.gov"
            
            response = requests.get(url, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return APIStatus(
                    name='USGS Landsat',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='USGS Landsat',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='USGS Landsat',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_nasa_modis(self) -> APIStatus:
        """Check NASA MODIS API (real satellite data)"""
        try:
            start_time = datetime.now()
            
            # Test the actual MODIS API endpoint
            url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
            params = {
                'latitude': 28.6139,  # Delhi coordinates for test
                'longitude': 77.2090,
                'startDate': '2024-01-01',
                'endDate': '2024-01-16',
                'band': '250m_16_days_NDVI',
                'kmAboveBelow': 0,
                'kmLeftRight': 0
            }
            
            response = requests.get(url, params=params, timeout=15)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200 and len(response.text) > 100:
                return APIStatus(
                    name='NASA MODIS',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='NASA MODIS',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}: Invalid response"
            )
            
        except Exception as e:
            return APIStatus(
                name='NASA MODIS',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_nasa_earthdata(self) -> APIStatus:
        """Check NASA EarthData accessibility"""
        try:
            start_time = datetime.now()
            
            url = "https://earthdata.nasa.gov"
            response = requests.get(url, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return APIStatus(
                    name='NASA EarthData',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='NASA EarthData',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='NASA EarthData',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_usgs_landsat(self) -> APIStatus:
        """Check USGS Landsat accessibility"""
        try:
            start_time = datetime.now()
            
            url = "https://landsatlook.usgs.gov"
            response = requests.get(url, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return APIStatus(
                    name='USGS Landsat',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='USGS Landsat',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='USGS Landsat',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_data_gov_in(self) -> APIStatus:
        """Check data.gov.in API"""
        try:
            start_time = datetime.now()
            
            url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
            params = {
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return APIStatus(
                    name='data.gov.in',
                    status='online',
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
            
            return APIStatus(
                name='data.gov.in',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='data.gov.in',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_openai(self) -> APIStatus:
        """Check OpenAI API"""
        try:
            start_time = datetime.now()
            
            url = "https://api.openai.com/v1/models"
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return APIStatus(
                        name='OpenAI',
                        status='online',
                        response_time_ms=response_time,
                        last_checked=datetime.now()
                    )
            
            return APIStatus(
                name='OpenAI',
                status='error',
                response_time_ms=response_time,
                last_checked=datetime.now(),
                error_message=f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            return APIStatus(
                name='OpenAI',
                status='offline',
                response_time_ms=0,
                last_checked=datetime.now(),
                error_message=str(e)
            )
    
    def generate_status_report(self, api_statuses: Dict[str, APIStatus]) -> str:
        """Generate a human-readable status report"""
        report = "🔍 AgriBotX Pro API Status Report\n"
        report += "=" * 50 + "\n\n"
        
        online_count = sum(1 for status in api_statuses.values() if status.status == 'online')
        total_count = len(api_statuses)
        
        report += f"Overall Status: {online_count}/{total_count} APIs online\n\n"
        
        # Group by category
        categories = {
            'Weather APIs': ['openweather', 'openmeteo', 'nasa_power'],
            'Satellite APIs (REAL DATA)': ['nasa_modis', 'nasa_earthdata', 'usgs_landsat'],
            'Market APIs': ['agmarknet_scraper', 'data_gov_in'],
            'AI APIs': ['openai']
        }
        
        for category, api_names in categories.items():
            report += f"{category}:\n"
            for api_name in api_names:
                if api_name in api_statuses:
                    status = api_statuses[api_name]
                    status_icon = {
                        'online': '✅',
                        'offline': '❌',
                        'error': '⚠️',
                        'no_key': '🔑'
                    }.get(status.status, '❓')
                    
                    report += f"  {status_icon} {status.name}: {status.status.upper()}"
                    
                    if status.status == 'online':
                        report += f" ({status.response_time_ms:.0f}ms)"
                    elif status.error_message:
                        report += f" - {status.error_message[:50]}..."
                    
                    report += "\n"
            report += "\n"
        
        # Recommendations
        report += "Recommendations:\n"
        
        no_key_apis = [s for s in api_statuses.values() if s.status == 'no_key']
        if no_key_apis:
            report += f"• Configure API keys for: {', '.join(s.name for s in no_key_apis)}\n"
        
        offline_apis = [s for s in api_statuses.values() if s.status == 'offline']
        if offline_apis:
            report += f"• Check network connectivity for: {', '.join(s.name for s in offline_apis)}\n"
        
        error_apis = [s for s in api_statuses.values() if s.status == 'error']
        if error_apis:
            report += f"• Investigate errors for: {', '.join(s.name for s in error_apis)}\n"
        
        if online_count == total_count:
            report += "• All APIs are operational! 🎉\n"
        
        return report

async def main():
    """Test the API status checker"""
    checker = APIStatusChecker()
    statuses = await checker.check_all_apis()
    report = checker.generate_status_report(statuses)
    print(report)

if __name__ == "__main__":
    asyncio.run(main())