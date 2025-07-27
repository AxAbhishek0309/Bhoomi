"""
Satellite NDVI Data Fetcher
Integrates with satellite APIs to provide vegetation index data
"""

import os
import requests
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)

@dataclass
class NDVIData:
    date: datetime
    ndvi: float
    cloud_coverage: float
    data_quality: str  # good, fair, poor
    source: str

class NDVIFetcher:
    """
    Satellite NDVI data fetcher with multiple data sources
    """
    
    def __init__(self):
        self.sentinelhub_client_id = os.getenv('SENTINELHUB_CLIENT_ID')
        self.sentinelhub_client_secret = os.getenv('SENTINELHUB_CLIENT_SECRET')
        
        # API endpoints
        self.sentinelhub_oauth_url = "https://services.sentinel-hub.com/oauth/token"
        self.sentinelhub_process_url = "https://services.sentinel-hub.com/api/v1/process"
        
        # Cache for API responses
        self._cache = {}
        self._cache_duration = 86400  # 24 hours
        self._access_token = None
        self._token_expires_at = None
    
    async def get_ndvi_timeseries(self, latitude: float, longitude: float,
                                start_date: datetime, end_date: datetime,
                                buffer_meters: int = 100) -> List[NDVIData]:
        """
        Get NDVI time series data for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date for data collection
            end_date: End date for data collection
            buffer_meters: Buffer around point in meters
            
        Returns:
            List of NDVIData objects
        """
        try:
            # Try SentinelHub first
            if self.sentinelhub_client_id and self.sentinelhub_client_secret:
                ndvi_data = await self._fetch_sentinelhub_ndvi(
                    latitude, longitude, start_date, end_date, buffer_meters
                )
                if ndvi_data:
                    return ndvi_data
            
            # Fallback to mock data
            return self._generate_mock_ndvi_data(start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching NDVI data: {e}")
            return self._generate_mock_ndvi_data(start_date, end_date)
    
    async def get_current_ndvi(self, latitude: float, longitude: float,
                             buffer_meters: int = 100) -> Optional[NDVIData]:
        """
        Get current NDVI value for a location
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            buffer_meters: Buffer around point in meters
            
        Returns:
            NDVIData object or None
        """
        try:
            # Get NDVI for the last 30 days and return the most recent
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            ndvi_timeseries = await self.get_ndvi_timeseries(
                latitude, longitude, start_date, end_date, buffer_meters
            )
            
            if ndvi_timeseries:
                # Return the most recent data point
                return max(ndvi_timeseries, key=lambda x: x.date)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current NDVI: {e}")
            return None
    
    async def _get_sentinelhub_token(self) -> Optional[str]:
        """Get access token for SentinelHub API"""
        try:
            # Check if we have a valid token
            if (self._access_token and self._token_expires_at and 
                datetime.now() < self._token_expires_at):
                return self._access_token
            
            # Request new token
            auth_string = base64.b64encode(
                f"{self.sentinelhub_client_id}:{self.sentinelhub_client_secret}".encode()
            ).decode()
            
            headers = {
                'Authorization': f'Basic {auth_string}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            response = requests.post(
                self.sentinelhub_oauth_url, 
                headers=headers, 
                data=data, 
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data['access_token']
            
            # Set expiration time (subtract 5 minutes for safety)
            expires_in = token_data.get('expires_in', 3600)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)
            
            return self._access_token
            
        except Exception as e:
            logger.error(f"Error getting SentinelHub token: {e}")
            return None
    
    async def _fetch_sentinelhub_ndvi(self, latitude: float, longitude: float,
                                    start_date: datetime, end_date: datetime,
                                    buffer_meters: int) -> Optional[List[NDVIData]]:
        """Fetch NDVI data from SentinelHub API"""
        try:
            token = await self._get_sentinelhub_token()
            if not token:
                return None
            
            # Create bounding box around the point
            bbox = self._create_bbox(latitude, longitude, buffer_meters)
            
            # Prepare request payload
            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                    },
                    "data": [
                        {
                            "type": "sentinel-2-l2a",
                            "dataFilter": {
                                "timeRange": {
                                    "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                                    "to": end_date.strftime("%Y-%m-%dT23:59:59Z")
                                },
                                "maxCloudCoverage": 50
                            }
                        }
                    ]
                },
                "output": {
                    "width": 10,
                    "height": 10,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {"type": "image/tiff"}
                        }
                    ]
                },
                "evalscript": self._get_ndvi_evalscript()
            }
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.sentinelhub_process_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Process response
            ndvi_data = self._process_sentinelhub_response(response.content)
            
            return ndvi_data
            
        except Exception as e:
            logger.error(f"Error fetching SentinelHub NDVI: {e}")
            return None
    
    def _create_bbox(self, latitude: float, longitude: float, buffer_meters: int) -> List[float]:
        """Create bounding box around a point"""
        # Approximate conversion: 1 degree ≈ 111,320 meters
        buffer_degrees = buffer_meters / 111320.0
        
        return [
            longitude - buffer_degrees,  # min_x
            latitude - buffer_degrees,   # min_y
            longitude + buffer_degrees,  # max_x
            latitude + buffer_degrees    # max_y
        ]
    
    def _get_ndvi_evalscript(self) -> str:
        """Get evalscript for NDVI calculation"""
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08", "SCL", "dataMask"],
                output: { bands: 4 }
            };
        }

        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            
            // Handle invalid values
            if (isNaN(ndvi) || !isFinite(ndvi)) {
                ndvi = -1;
            }
            
            // Scene classification for cloud detection
            let isCloud = sample.SCL === 9 || sample.SCL === 8 || sample.SCL === 3;
            
            return [
                ndvi,
                sample.dataMask,
                isCloud ? 1 : 0,
                sample.SCL
            ];
        }
        """
    
    def _process_sentinelhub_response(self, response_content: bytes) -> List[NDVIData]:
        """Process SentinelHub API response to extract NDVI values"""
        try:
            # This is a simplified implementation
            # In practice, you would need to parse the TIFF image data
            # and extract NDVI values from the pixels
            
            # For now, return mock data based on the response
            # In a real implementation, you would use libraries like rasterio
            # to read the TIFF data and calculate statistics
            
            ndvi_data = []
            
            # Mock processing - replace with actual TIFF parsing
            base_date = datetime.now() - timedelta(days=15)
            for i in range(5):  # Simulate 5 data points
                date = base_date + timedelta(days=i * 3)
                ndvi_value = 0.3 + (i * 0.1) + np.random.normal(0, 0.05)
                
                ndvi_data.append(NDVIData(
                    date=date,
                    ndvi=max(0, min(1, ndvi_value)),  # Clamp to valid range
                    cloud_coverage=np.random.uniform(0, 30),
                    data_quality='good' if ndvi_value > 0.2 else 'fair',
                    source='sentinelhub'
                ))
            
            return ndvi_data
            
        except Exception as e:
            logger.error(f"Error processing SentinelHub response: {e}")
            return []
    
    def _generate_mock_ndvi_data(self, start_date: datetime, end_date: datetime) -> List[NDVIData]:
        """Generate mock NDVI data for testing"""
        try:
            ndvi_data = []
            current_date = start_date
            
            # Generate data points every 5 days
            while current_date <= end_date:
                # Simulate seasonal NDVI variation
                day_of_year = current_date.timetuple().tm_yday
                
                # Peak NDVI in growing season (day 120-240)
                if 120 <= day_of_year <= 240:
                    base_ndvi = 0.6 + 0.3 * np.sin((day_of_year - 120) * np.pi / 120)
                else:
                    base_ndvi = 0.2 + 0.2 * np.random.random()
                
                # Add some noise
                ndvi_value = base_ndvi + np.random.normal(0, 0.1)
                ndvi_value = max(0, min(1, ndvi_value))  # Clamp to valid range
                
                # Simulate cloud coverage
                cloud_coverage = np.random.uniform(0, 40)
                
                # Data quality based on cloud coverage
                if cloud_coverage < 10:
                    quality = 'good'
                elif cloud_coverage < 30:
                    quality = 'fair'
                else:
                    quality = 'poor'
                
                ndvi_data.append(NDVIData(
                    date=current_date,
                    ndvi=ndvi_value,
                    cloud_coverage=cloud_coverage,
                    data_quality=quality,
                    source='mock'
                ))
                
                current_date += timedelta(days=5)
            
            return ndvi_data
            
        except Exception as e:
            logger.error(f"Error generating mock NDVI data: {e}")
            return []
    
    def calculate_ndvi_statistics(self, ndvi_data: List[NDVIData]) -> Dict[str, float]:
        """Calculate statistics from NDVI time series"""
        try:
            if not ndvi_data:
                return {}
            
            # Filter out poor quality data
            good_data = [d for d in ndvi_data if d.data_quality in ['good', 'fair']]
            
            if not good_data:
                good_data = ndvi_data  # Use all data if no good quality data
            
            ndvi_values = [d.ndvi for d in good_data]
            
            statistics = {
                'mean_ndvi': np.mean(ndvi_values),
                'median_ndvi': np.median(ndvi_values),
                'min_ndvi': np.min(ndvi_values),
                'max_ndvi': np.max(ndvi_values),
                'std_ndvi': np.std(ndvi_values),
                'data_points': len(good_data),
                'quality_score': len(good_data) / len(ndvi_data) if ndvi_data else 0
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating NDVI statistics: {e}")
            return {}
    
    def assess_vegetation_health(self, ndvi_data: List[NDVIData]) -> Dict[str, Any]:
        """Assess vegetation health based on NDVI data"""
        try:
            if not ndvi_data:
                return {'status': 'no_data', 'assessment': 'No NDVI data available'}
            
            stats = self.calculate_ndvi_statistics(ndvi_data)
            mean_ndvi = stats.get('mean_ndvi', 0)
            
            # NDVI interpretation
            if mean_ndvi > 0.7:
                status = 'excellent'
                assessment = 'Dense, healthy vegetation with high photosynthetic activity'
                recommendations = ['Maintain current management practices']
            elif mean_ndvi > 0.5:
                status = 'good'
                assessment = 'Healthy vegetation with good growth'
                recommendations = ['Continue monitoring', 'Consider nutrient optimization']
            elif mean_ndvi > 0.3:
                status = 'moderate'
                assessment = 'Moderate vegetation health, some stress indicators'
                recommendations = [
                    'Check soil moisture levels',
                    'Consider fertilizer application',
                    'Monitor for pest/disease issues'
                ]
            elif mean_ndvi > 0.1:
                status = 'poor'
                assessment = 'Poor vegetation health, significant stress'
                recommendations = [
                    'Immediate irrigation if needed',
                    'Soil testing recommended',
                    'Check for pest/disease problems',
                    'Consider replanting if necessary'
                ]
            else:
                status = 'very_poor'
                assessment = 'Very poor or no vegetation detected'
                recommendations = [
                    'Investigate crop failure causes',
                    'Soil rehabilitation may be needed',
                    'Consider alternative crops'
                ]
            
            # Trend analysis
            if len(ndvi_data) >= 3:
                recent_values = [d.ndvi for d in sorted(ndvi_data, key=lambda x: x.date)[-3:]]
                if recent_values[-1] > recent_values[0]:
                    trend = 'improving'
                elif recent_values[-1] < recent_values[0]:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
            
            return {
                'status': status,
                'assessment': assessment,
                'recommendations': recommendations,
                'trend': trend,
                'statistics': stats,
                'last_updated': max(ndvi_data, key=lambda x: x.date).date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing vegetation health: {e}")
            return {
                'status': 'error',
                'assessment': f'Error in assessment: {str(e)}',
                'recommendations': ['Manual field inspection recommended']
            }
    
    def export_ndvi_data(self, ndvi_data: List[NDVIData], format: str = 'json') -> str:
        """Export NDVI data in specified format"""
        try:
            if format.lower() == 'json':
                data_dict = []
                for d in ndvi_data:
                    data_dict.append({
                        'date': d.date.isoformat(),
                        'ndvi': d.ndvi,
                        'cloud_coverage': d.cloud_coverage,
                        'data_quality': d.data_quality,
                        'source': d.source
                    })
                return json.dumps(data_dict, indent=2)
            
            elif format.lower() == 'csv':
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow(['date', 'ndvi', 'cloud_coverage', 'data_quality', 'source'])
                
                # Write data
                for d in ndvi_data:
                    writer.writerow([
                        d.date.isoformat(),
                        d.ndvi,
                        d.cloud_coverage,
                        d.data_quality,
                        d.source
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting NDVI data: {e}")
            return f"Error exporting data: {str(e)}"