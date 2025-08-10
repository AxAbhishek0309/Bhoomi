#!/usr/bin/env python3
"""
Test NASA APIs for real satellite NDVI data
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.satellite_ndvi import NDVIFetcher

async def test_nasa_apis():
    """Test all NASA satellite APIs"""
    print("🛰️  Testing NASA Satellite APIs for Real NDVI Data")
    print("=" * 60)
    
    # Initialize NDVI fetcher
    fetcher = NDVIFetcher()
    
    # Test coordinates (Delhi, India - agricultural area)
    latitude = 28.6139
    longitude = 77.2090
    
    # Test date range (last 60 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print(f"📍 Location: {latitude}°N, {longitude}°E (Delhi, India)")
    print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    # Test API connectivity
    print("🔍 Testing API Connectivity...")
    api_status = await fetcher.test_nasa_apis(latitude, longitude)
    
    for api_name, is_online in api_status.items():
        status_icon = "✅" if is_online else "❌"
        print(f"   {status_icon} {api_name}: {'Online' if is_online else 'Offline'}")
    
    print()
    
    # Test NDVI data fetching
    print("📊 Fetching Real NDVI Data...")
    
    try:
        ndvi_data = await fetcher.get_ndvi_timeseries(
            latitude, longitude, start_date, end_date
        )
        
        if ndvi_data and len(ndvi_data) > 0:
            print(f"✅ Successfully fetched {len(ndvi_data)} NDVI data points!")
            print()
            
            # Display statistics
            stats = fetcher.calculate_ndvi_statistics(ndvi_data)
            
            print("📈 NDVI Statistics:")
            print(f"   Mean NDVI: {stats.get('mean_ndvi', 0):.3f}")
            print(f"   Min NDVI:  {stats.get('min_ndvi', 0):.3f}")
            print(f"   Max NDVI:  {stats.get('max_ndvi', 0):.3f}")
            print(f"   Std Dev:   {stats.get('std_ndvi', 0):.3f}")
            print(f"   Data Points: {stats.get('data_points', 0)}")
            print()
            
            # Display sample data points
            print("📋 Sample NDVI Data Points:")
            for i, data_point in enumerate(ndvi_data[:5]):  # Show first 5
                print(f"   {i+1}. {data_point.date.strftime('%Y-%m-%d')}: "
                      f"NDVI={data_point.ndvi:.3f}, "
                      f"Quality={data_point.data_quality}, "
                      f"Source={data_point.source}")
            
            if len(ndvi_data) > 5:
                print(f"   ... and {len(ndvi_data) - 5} more data points")
            
            print()
            
            # Assess vegetation health
            health_assessment = fetcher.assess_vegetation_health(ndvi_data)
            
            print("🌱 Vegetation Health Assessment:")
            print(f"   Status: {health_assessment.get('status', 'unknown').title()}")
            print(f"   Assessment: {health_assessment.get('assessment', 'N/A')}")
            print(f"   Trend: {health_assessment.get('trend', 'unknown').title()}")
            
            if 'recommendations' in health_assessment:
                print("   Recommendations:")
                for rec in health_assessment['recommendations'][:3]:
                    print(f"     • {rec}")
            
        else:
            print("❌ No NDVI data retrieved")
            print("   This could be due to:")
            print("   • Network connectivity issues")
            print("   • NASA API temporary unavailability")
            print("   • No satellite coverage for the specified area/time")
    
    except Exception as e:
        print(f"❌ Error fetching NDVI data: {e}")
    
    print()
    print("=" * 60)
    print("🎯 NASA API Test Summary:")
    
    online_apis = sum(1 for status in api_status.values() if status)
    total_apis = len(api_status)
    
    if online_apis == total_apis:
        print("✅ All NASA APIs are accessible!")
    elif online_apis > 0:
        print(f"⚠️  {online_apis}/{total_apis} NASA APIs are accessible")
    else:
        print("❌ No NASA APIs are accessible - check internet connection")
    
    print()
    print("💡 About NASA Satellite Data:")
    print("• MODIS: 250m resolution, 16-day composites, free")
    print("• Landsat: 30m resolution, 16-day revisit, free")
    print("• VIIRS: 500m resolution, daily, free")
    print("• All data is from real NASA satellites! 🛰️")

if __name__ == "__main__":
    asyncio.run(test_nasa_apis())