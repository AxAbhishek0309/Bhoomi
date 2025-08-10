#!/usr/bin/env python3
"""
Test script for the simplified AgriBotX Pro system
Tests all components without requiring complex API setups
"""

import os
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

async def test_weather_apis():
    """Test weather data fetching"""
    print("🌤️  Testing Weather APIs...")
    
    try:
        from utils.weather_fetcher import WeatherFetcher
        
        fetcher = WeatherFetcher()
        
        # Test coordinates (Delhi)
        lat, lon = 28.6139, 77.2090
        
        # Test current weather
        try:
            weather = await fetcher.get_current_weather(lat, lon)
            if weather and weather.get('temperature'):
                print(f"   ✅ Current weather: {weather['temperature']}°C from {weather['source']}")
            else:
                print("   ❌ No current weather data")
        except Exception as e:
            print(f"   ❌ Weather error: {e}")
        
        # Test forecast
        try:
            forecast = await fetcher.get_forecast(lat, lon, 3)
            if forecast and forecast.get('daily'):
                print(f"   ✅ Forecast: {len(forecast['daily'])} days from {forecast['source']}")
            else:
                print("   ❌ No forecast data")
        except Exception as e:
            print(f"   ❌ Forecast error: {e}")
            
    except Exception as e:
        print(f"   ❌ Weather module error: {e}")

async def test_satellite_apis():
    """Test satellite NDVI data"""
    print("\n🛰️  Testing Satellite APIs...")
    
    try:
        from utils.satellite_ndvi import NDVIFetcher
        from datetime import datetime, timedelta
        
        fetcher = NDVIFetcher()
        
        # Test coordinates (Delhi)
        lat, lon = 28.6139, 77.2090
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        try:
            ndvi_data = await fetcher.get_ndvi_timeseries(lat, lon, start_date, end_date)
            if ndvi_data and len(ndvi_data) > 0:
                avg_ndvi = sum(d.ndvi for d in ndvi_data) / len(ndvi_data)
                print(f"   ✅ NDVI data: {len(ndvi_data)} points, avg NDVI: {avg_ndvi:.3f}")
                print(f"   📊 Source: {ndvi_data[0].source}")
            else:
                print("   ❌ No NDVI data")
        except Exception as e:
            print(f"   ❌ NDVI error: {e}")
            
    except Exception as e:
        print(f"   ❌ Satellite module error: {e}")

async def test_market_scraper():
    """Test market price scraping"""
    print("\n💰 Testing Market Price Scraper...")
    
    try:
        from utils.agmarknet_scraper import AgmarknetScraper
        
        scraper = AgmarknetScraper()
        
        # Test connection
        if scraper.test_connection():
            print("   ✅ Agmarknet website accessible")
        else:
            print("   ❌ Agmarknet website not accessible")
            return
        
        # Test scraping
        test_crops = ['wheat', 'rice']
        
        for crop in test_crops:
            try:
                prices = await scraper.get_crop_prices(crop)
                if prices and len(prices) > 0:
                    avg_price = sum(p['modal_price'] for p in prices if p['modal_price'] > 0) / len([p for p in prices if p['modal_price'] > 0])
                    print(f"   ✅ {crop.title()}: {len(prices)} markets, avg price: ₹{avg_price:.0f}/quintal")
                else:
                    print(f"   ⚠️  {crop.title()}: No prices found (website structure may have changed)")
            except Exception as e:
                print(f"   ❌ {crop.title()} scraping error: {e}")
                
    except Exception as e:
        print(f"   ❌ Market scraper error: {e}")

async def test_api_status():
    """Test API status checker"""
    print("\n🔍 Testing API Status Checker...")
    
    try:
        from utils.api_status import APIStatusChecker
        
        checker = APIStatusChecker()
        statuses = await checker.check_all_apis()
        
        online_count = sum(1 for s in statuses.values() if s.status == 'online')
        total_count = len(statuses)
        
        print(f"   📊 API Status: {online_count}/{total_count} online")
        
        for name, status in statuses.items():
            status_icon = {
                'online': '✅',
                'offline': '❌',
                'error': '⚠️',
                'no_key': '🔑'
            }.get(status.status, '❓')
            
            print(f"   {status_icon} {status.name}: {status.status}")
            
    except Exception as e:
        print(f"   ❌ API status checker error: {e}")

def test_environment_setup():
    """Test environment configuration"""
    print("🔧 Testing Environment Setup...")
    
    # Check .env.example
    env_example_path = Path('.env.example')
    if env_example_path.exists():
        print("   ✅ .env.example exists")
        
        with open(env_example_path, 'r') as f:
            content = f.read()
            
        if 'OPENAI_API_KEY' in content:
            print("   ✅ OpenAI API key placeholder found")
        else:
            print("   ❌ OpenAI API key placeholder missing")
            
        if 'OPENWEATHER_API_KEY' in content:
            print("   ✅ OpenWeather API key placeholder found")
        else:
            print("   ❌ OpenWeather API key placeholder missing")
            
        # Check that complex APIs are removed
        if 'SENTINELHUB' not in content:
            print("   ✅ SentinelHub removed (good!)")
        else:
            print("   ⚠️  SentinelHub still present")
            
        if 'TWILIO' not in content:
            print("   ✅ Twilio removed (good!)")
        else:
            print("   ⚠️  Twilio still present")
            
    else:
        print("   ❌ .env.example not found")

def test_requirements():
    """Test requirements.txt"""
    print("\n📦 Testing Requirements...")
    
    req_path = Path('requirements.txt')
    if req_path.exists():
        print("   ✅ requirements.txt exists")
        
        with open(req_path, 'r') as f:
            content = f.read()
            
        if 'beautifulsoup4' in content:
            print("   ✅ BeautifulSoup4 added for web scraping")
        else:
            print("   ❌ BeautifulSoup4 missing")
            
        if 'twilio' not in content:
            print("   ✅ Twilio removed (good!)")
        else:
            print("   ⚠️  Twilio still in requirements")
            
    else:
        print("   ❌ requirements.txt not found")

async def main():
    """Run all tests"""
    print("🧪 AgriBotX Pro Simplified System Test")
    print("=" * 50)
    
    # Test environment setup
    test_environment_setup()
    test_requirements()
    
    # Test API components
    await test_weather_apis()
    await test_satellite_apis()
    await test_market_scraper()
    await test_api_status()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    print("\n💡 Next steps:")
    print("1. Add your OpenAI API key to .env")
    print("2. Optionally add OpenWeatherMap API key")
    print("3. Run: python main.py --mode check-apis")
    print("4. Run: python main.py --mode web")

if __name__ == "__main__":
    asyncio.run(main())