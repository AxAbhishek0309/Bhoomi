"""
Spatial query utilities for AgriBotX Pro
Handles geospatial operations using PostGIS
"""

from sqlalchemy import text, func
from sqlalchemy.orm import Session
from geoalchemy2.functions import ST_DWithin, ST_Distance, ST_GeomFromText, ST_AsText
from db.models import Farm, WeatherData
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def create_spatial_indexes(engine):
    """Create spatial indexes for better query performance"""
    with engine.connect() as conn:
        try:
            # Create spatial indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_farms_location ON farms USING GIST (location)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_weather_location ON weather_data USING GIST (location)"))
            
            # Create composite indexes for common queries
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_market_prices_crop_date ON market_prices (crop_name, price_date DESC)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_soil_reports_farm_date ON soil_reports (farm_id, report_date DESC)"))
            
            conn.commit()
            logger.info("Spatial indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating spatial indexes: {e}")
            conn.rollback()

def find_nearby_farms(session: Session, latitude: float, longitude: float, 
                     radius_km: float = 10) -> List[Farm]:
    """
    Find farms within a specified radius of a given point
    
    Args:
        session: SQLAlchemy session
        latitude: Latitude of the center point
        longitude: Longitude of the center point
        radius_km: Search radius in kilometers
    
    Returns:
        List of Farm objects within the radius
    """
    try:
        point = f"POINT({longitude} {latitude})"
        
        farms = session.query(Farm).filter(
            ST_DWithin(
                Farm.location,
                ST_GeomFromText(point, 4326),
                radius_km * 1000  # Convert km to meters
            )
        ).all()
        
        return farms
        
    except Exception as e:
        logger.error(f"Error finding nearby farms: {e}")
        return []

def get_nearest_weather_station(session: Session, farm_id: str) -> Optional[WeatherData]:
    """
    Find the nearest weather station to a given farm
    
    Args:
        session: SQLAlchemy session
        farm_id: UUID of the farm
    
    Returns:
        WeatherData object of the nearest station or None
    """
    try:
        farm = session.query(Farm).filter(Farm.id == farm_id).first()
        if not farm:
            return None
        
        nearest_weather = session.query(WeatherData).order_by(
            ST_Distance(WeatherData.location, farm.location)
        ).first()
        
        return nearest_weather
        
    except Exception as e:
        logger.error(f"Error finding nearest weather station: {e}")
        return None

def calculate_farm_distance(session: Session, farm1_id: str, farm2_id: str) -> Optional[float]:
    """
    Calculate distance between two farms in kilometers
    
    Args:
        session: SQLAlchemy session
        farm1_id: UUID of first farm
        farm2_id: UUID of second farm
    
    Returns:
        Distance in kilometers or None if error
    """
    try:
        farm1 = session.query(Farm).filter(Farm.id == farm1_id).first()
        farm2 = session.query(Farm).filter(Farm.id == farm2_id).first()
        
        if not farm1 or not farm2:
            return None
        
        distance = session.query(
            ST_Distance(farm1.location, farm2.location)
        ).scalar()
        
        # Convert from degrees to kilometers (approximate)
        return distance * 111.32  # 1 degree ≈ 111.32 km
        
    except Exception as e:
        logger.error(f"Error calculating farm distance: {e}")
        return None

def get_farms_in_region(session: Session, region_name: str, 
                       state: str = None) -> List[Farm]:
    """
    Get all farms in a specific administrative region
    
    Args:
        session: SQLAlchemy session
        region_name: Name of the region/district
        state: Optional state filter
    
    Returns:
        List of Farm objects in the region
    """
    try:
        # This would typically use administrative boundary data
        # For now, we'll use a simple text search approach
        query = session.query(Farm)
        
        # In a production system, you would join with administrative boundary tables
        # and use spatial intersection queries
        
        # Placeholder implementation - would need actual boundary data
        farms = query.all()  # Return all farms for now
        
        return farms
        
    except Exception as e:
        logger.error(f"Error getting farms in region: {e}")
        return []

def create_farm_buffer_zone(session: Session, farm_id: str, 
                           buffer_km: float = 1.0) -> str:
    """
    Create a buffer zone around a farm for analysis
    
    Args:
        session: SQLAlchemy session
        farm_id: UUID of the farm
        buffer_km: Buffer distance in kilometers
    
    Returns:
        WKT string of the buffer geometry
    """
    try:
        farm = session.query(Farm).filter(Farm.id == farm_id).first()
        if not farm:
            return None
        
        # Create buffer in meters (buffer_km * 1000)
        buffer_geom = session.query(
            ST_AsText(func.ST_Buffer(farm.location, buffer_km * 1000))
        ).scalar()
        
        return buffer_geom
        
    except Exception as e:
        logger.error(f"Error creating farm buffer zone: {e}")
        return None

def get_regional_weather_summary(session: Session, latitude: float, longitude: float,
                                radius_km: float = 50) -> dict:
    """
    Get weather summary for a region around a point
    
    Args:
        session: SQLAlchemy session
        latitude: Center latitude
        longitude: Center longitude
        radius_km: Analysis radius in kilometers
    
    Returns:
        Dictionary with weather statistics
    """
    try:
        point = f"POINT({longitude} {latitude})"
        
        weather_data = session.query(WeatherData).filter(
            ST_DWithin(
                WeatherData.location,
                ST_GeomFromText(point, 4326),
                radius_km * 1000
            )
        ).all()
        
        if not weather_data:
            return {}
        
        # Calculate regional statistics
        temperatures = [w.temperature_celsius for w in weather_data if w.temperature_celsius]
        rainfall = [w.rainfall_mm for w in weather_data if w.rainfall_mm]
        humidity = [w.humidity_percent for w in weather_data if w.humidity_percent]
        
        summary = {
            'station_count': len(weather_data),
            'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else None,
            'total_rainfall': sum(rainfall) if rainfall else None,
            'avg_humidity': sum(humidity) / len(humidity) if humidity else None,
            'coverage_area_km2': 3.14159 * (radius_km ** 2)
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting regional weather summary: {e}")
        return {}

def find_optimal_farm_locations(session: Session, crop_type: str, 
                               region_bounds: Tuple[float, float, float, float],
                               max_results: int = 10) -> List[dict]:
    """
    Find optimal locations for a specific crop within a region
    
    Args:
        session: SQLAlchemy session
        crop_type: Type of crop to optimize for
        region_bounds: (min_lat, min_lon, max_lat, max_lon)
        max_results: Maximum number of results to return
    
    Returns:
        List of dictionaries with location and suitability scores
    """
    try:
        min_lat, min_lon, max_lat, max_lon = region_bounds
        
        # This would typically involve complex spatial analysis
        # combining soil data, weather patterns, market access, etc.
        
        # Placeholder implementation
        optimal_locations = []
        
        # In a real implementation, this would:
        # 1. Create a grid of potential locations
        # 2. Score each location based on multiple factors
        # 3. Return the top-scoring locations
        
        return optimal_locations
        
    except Exception as e:
        logger.error(f"Error finding optimal farm locations: {e}")
        return []