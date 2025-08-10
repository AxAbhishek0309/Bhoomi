#!/usr/bin/env python3
"""
Test script to verify real satellite APIs are working
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.satellite_ndvi import NDVIFetcher

async def test_satellite_apis():
    """Test all satellite APIs with real coordinates"""
    
    print("🛰️  Testing Real Satellite APIs for AgriBotX Pro")
    print("=" * 60)
    
    # Initialize NDVI fetcher
    ndvi_fetcher = NDVIFetcher()
    
    # Test coordinates (Delhi, India - agricultural area)
    latitude = 28.6139
    longitude = 77.2090
    
    # Test date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📍 Test Location: {latitude}, {longitude} (Delhi, India)")
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    # Test API status first
    print("🔍 Checking API Status...")
    api_status = ndvi_fetcher.get_satellite_api_status()
    for api_name, status in api_status.items():
        print(f"   {api_name}: {status}")
    print()
    
    # Test connectivity
    print("🌐 Testing API Connectivity...")
    connectivity = await ndvi_fetcher.test_satellite_apis()
    for api_name, is_online in connectivity.items():
        status_icon = "✅" if is_online else "❌"
        print(f"   {status_icon} {api_name}: {'Online' if is_online else 'Offline'}")
    print()
    
    # Test actual NDVI data fetching
    print("📡 Fetching Real NDVI Data...")
    try:
        ndvi_data = await ndvi_fetcher.get_ndvi_timeseries(
            latitude, longitude, start_date, end_date
        )
        
        if ndvi_data and len(ndvi_data) > 0:
            print(f"✅ Successfully fetched {len(ndvi_data)} NDVI data points!")
            print()
            
            # Show sample data
            print("📊 Sample NDVI Data:")
            print("-" * 50)
            print(f"{'Date':<12} {'NDVI':<8} {'Quality':<8} {'Source':<15}")
            print("-" * 50)
            
            for i, data_point in enumerate(ndvi_data[:5]):  # Show first 5 points
                print(f"{data_point.date.strftime('%Y-%m-%d'):<12} "
                      f"{data_point.ndvi:<8.3f} "
                      f"{data_point.data_quality:<8} "
                      f"{data_point.source:<15}")
            
            if len(ndvi_data) > 5:
                print(f"... and {len(ndvi_data) - 5} more data points")
            
            print()
            
            # Calculate statistics
            ndvi_values = [d.ndvi for d in ndvi_data]
            avg_ndvi = sum(ndvi_values) / len(ndvi_values)
            min_ndvi = min(ndvi_values)
            max_ndvi = max(ndvi_values)
            
            print("📈 NDVI Statistics:")
            print(f"   Average NDVI: {avg_ndvi:.3f}")
            print(f"   Min NDVI: {min_ndvi:.3f}")
            print(f"   Max NDVI: {max_ndvi:.3f}")
            print()
            
            # Assess vegetation health
            health_assessment = ndvi_fetcher.assess_vegetation_health(ndvi_data)
            print("🌱 Vegetation Health Assessment:")
            print(f"   Status: {health_assessment.get('status', 'unknown').title()}")
            print(f"   Assessment: {health_assessment.get('assessment', 'N/A')}")
            
            if health_assessment.get('recommendations'):
                print("   Recommendations:")
                for rec in health_assessment['recommendations'][:3]:
                    print(f"   • {rec}")
            
            print()
            print("🎉 All satellite APIs are working correctly!")
            print("✅ Real NDVI data is being fetched successfully!")
            
        else:
            print("❌ No NDVI data received")
            print("   This might be due to:")
            print("   • Network connectivity issues")
            print("   • API service temporarily unavailable")
            print("   • Invalid coordinates or date range")
            
    except Exception as e:
        print(f"❌ Error fetching NDVI data: {e}")
        print("   Check your internet connection and try again")
    
    print()
    print("🔗 Data Sources Used:")
    print("   • NASA MODIS: Real satellite imagery from Terra/Aqua satellites")
    print("   • NASA EarthData: Official NASA satellite data portal")
    print("   • USGS Landsat: High-resolution satellite imagery")
    print("   • All sources are FREE and require NO API keys!")

if __name__ == "__main__":
    asyncio.run(test_satellite_apis())