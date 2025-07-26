"""
Irrigation Advisor Agent
Provides intelligent irrigation scheduling based on weather, soil, and crop data
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from sqlalchemy.orm import Session
from db.models import Farm, SoilReport, WeatherData, IrrigationSchedule
from utils.weather_fetcher import WeatherFetcher

logger = logging.getLogger(__name__)

@dataclass
class IrrigationRecommendation:
    scheduled_date: datetime
    water_amount_liters: float
    irrigation_method: str
    duration_minutes: int
    reasoning: str
    urgency_level: str  # low, medium, high, critical
    weather_considerations: List[str]

@dataclass
class SoilMoistureStatus:
    current_level: float  # 0-1 scale
    optimal_range: Tuple[float, float]
    status: str  # dry, optimal, wet
    days_until_critical: Optional[int]

class IrrigationAdvisorAgent:
    """
    AI-powered irrigation advisor that optimizes water usage based on multiple factors
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weather_fetcher = WeatherFetcher()
        self.forecast_days = config.get('weather_forecast_days', 7)
        self.soil_moisture_threshold = config.get('soil_moisture_threshold', 0.3)
        
        # Crop water requirements (mm per season)
        self.crop_water_requirements = config.get('crop_water_requirements', {
            'rice': 1200,
            'wheat': 450,
            'corn': 600,
            'cotton': 700,
            'tomato': 400,
            'potato': 350,
            'sugarcane': 1800,
            'soybean': 500,
            'onion': 350,
            'cabbage': 380
        })
        
        # Crop coefficients for different growth stages
        self.crop_coefficients = {
            'rice': {'initial': 1.05, 'development': 1.20, 'mid': 1.20, 'late': 0.90},
            'wheat': {'initial': 0.40, 'development': 0.70, 'mid': 1.15, 'late': 0.40},
            'corn': {'initial': 0.30, 'development': 0.70, 'mid': 1.20, 'late': 0.60},
            'tomato': {'initial': 0.60, 'development': 0.80, 'mid': 1.15, 'late': 0.80},
            'potato': {'initial': 0.50, 'development': 0.75, 'mid': 1.15, 'late': 0.85}
        }
        
        # Soil type water holding capacity (mm/m depth)
        self.soil_water_capacity = {
            'sandy': 100,
            'loamy': 150,
            'clay': 200,
            'silt': 180
        }
    
    async def get_irrigation_recommendation(self, farm_id: str, crop_type: str, 
                                         growth_stage: str, session: Session) -> IrrigationRecommendation:
        """
        Generate irrigation recommendation for a specific farm and crop
        
        Args:
            farm_id: UUID of the farm
            crop_type: Type of crop being grown
            growth_stage: Current growth stage (initial, development, mid, late)
            session: Database session
            
        Returns:
            IrrigationRecommendation with detailed scheduling information
        """
        try:
            # Get farm information
            farm = session.query(Farm).filter(Farm.id == farm_id).first()
            if not farm:
                raise ValueError(f"Farm with ID {farm_id} not found")
            
            # Get soil moisture status
            soil_moisture = await self._assess_soil_moisture(farm_id, session)
            
            # Get weather forecast
            weather_forecast = await self._get_weather_forecast(farm)
            
            # Calculate crop water requirement
            daily_water_need = self._calculate_daily_water_requirement(
                crop_type, growth_stage, farm.area_hectares
            )
            
            # Analyze upcoming rainfall
            rainfall_forecast = self._analyze_rainfall_forecast(weather_forecast)
            
            # Determine irrigation need
            irrigation_need = self._determine_irrigation_need(
                soil_moisture, daily_water_need, rainfall_forecast
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                irrigation_need, soil_moisture, weather_forecast, 
                daily_water_need, farm.area_hectares
            )
            
            # Save to database
            await self._save_irrigation_schedule(farm_id, crop_type, recommendation, session)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating irrigation recommendation: {e}")
            return IrrigationRecommendation(
                scheduled_date=datetime.now() + timedelta(days=1),
                water_amount_liters=1000,
                irrigation_method="sprinkler",
                duration_minutes=60,
                reasoning=f"Error in calculation: {str(e)}",
                urgency_level="medium",
                weather_considerations=[]
            )
    
    async def _assess_soil_moisture(self, farm_id: str, session: Session) -> SoilMoistureStatus:
        """Assess current soil moisture status"""
        try:
            # Get latest soil report
            soil_report = session.query(SoilReport).filter(
                SoilReport.farm_id == farm_id
            ).order_by(SoilReport.report_date.desc()).first()
            
            if soil_report and soil_report.moisture_content:
                current_moisture = soil_report.moisture_content / 100.0  # Convert to 0-1 scale
            else:
                # Estimate based on recent weather if no soil data
                current_moisture = 0.4  # Default assumption
            
            # Determine optimal range based on soil type
            farm = session.query(Farm).filter(Farm.id == farm_id).first()
            soil_type = farm.soil_type.lower() if farm.soil_type else 'loamy'
            
            if soil_type == 'sandy':
                optimal_range = (0.15, 0.35)
            elif soil_type == 'clay':
                optimal_range = (0.25, 0.45)
            else:  # loamy or silt
                optimal_range = (0.20, 0.40)
            
            # Determine status
            if current_moisture < optimal_range[0]:
                status = 'dry'
                days_until_critical = max(1, int((optimal_range[0] - current_moisture) * 10))
            elif current_moisture > optimal_range[1]:
                status = 'wet'
                days_until_critical = None
            else:
                status = 'optimal'
                days_until_critical = None
            
            return SoilMoistureStatus(
                current_level=current_moisture,
                optimal_range=optimal_range,
                status=status,
                days_until_critical=days_until_critical
            )
            
        except Exception as e:
            logger.error(f"Error assessing soil moisture: {e}")
            return SoilMoistureStatus(
                current_level=0.3,
                optimal_range=(0.2, 0.4),
                status='unknown',
                days_until_critical=2
            )
    
    async def _get_weather_forecast(self, farm: Farm) -> Dict[str, Any]:
        """Get weather forecast for the farm location"""
        try:
            latitude = farm.location.y
            longitude = farm.location.x
            
            forecast = await self.weather_fetcher.get_forecast(
                latitude, longitude, days=self.forecast_days
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error getting weather forecast: {e}")
            return {}
    
    def _calculate_daily_water_requirement(self, crop_type: str, growth_stage: str, 
                                         area_hectares: float) -> float:
        """Calculate daily water requirement for the crop"""
        try:
            # Base water requirement
            base_requirement = self.crop_water_requirements.get(crop_type, 500)  # mm per season
            
            # Crop coefficient for growth stage
            crop_coeff = self.crop_coefficients.get(crop_type, {}).get(growth_stage, 1.0)
            
            # Daily requirement (assuming 120-day growing season)
            daily_mm = (base_requirement / 120) * crop_coeff
            
            # Convert to liters for the farm area
            daily_liters = daily_mm * area_hectares * 10  # 1mm over 1 hectare = 10,000 liters
            
            return daily_liters
            
        except Exception as e:
            logger.error(f"Error calculating water requirement: {e}")
            return 1000.0  # Default fallback
    
    def _analyze_rainfall_forecast(self, weather_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze upcoming rainfall from weather forecast"""
        try:
            if not weather_forecast or 'daily' not in weather_forecast:
                return {'total_expected': 0, 'days_with_rain': 0, 'heavy_rain_days': 0}
            
            daily_data = weather_forecast['daily']
            total_rainfall = 0
            days_with_rain = 0
            heavy_rain_days = 0
            
            for day in daily_data[:7]:  # Next 7 days
                rainfall = day.get('precipitation', 0)
                if rainfall > 0:
                    total_rainfall += rainfall
                    days_with_rain += 1
                    if rainfall > 10:  # Heavy rain threshold
                        heavy_rain_days += 1
            
            return {
                'total_expected': total_rainfall,
                'days_with_rain': days_with_rain,
                'heavy_rain_days': heavy_rain_days,
                'next_rain_day': self._find_next_rain_day(daily_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing rainfall forecast: {e}")
            return {'total_expected': 0, 'days_with_rain': 0, 'heavy_rain_days': 0}
    
    def _find_next_rain_day(self, daily_data: List[Dict]) -> Optional[int]:
        """Find the next day with significant rainfall"""
        for i, day in enumerate(daily_data):
            if day.get('precipitation', 0) > 5:  # 5mm threshold
                return i + 1
        return None
    
    def _determine_irrigation_need(self, soil_moisture: SoilMoistureStatus, 
                                 daily_water_need: float, 
                                 rainfall_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Determine irrigation need based on all factors"""
        try:
            need_score = 0
            reasons = []
            
            # Soil moisture factor
            if soil_moisture.status == 'dry':
                need_score += 40
                reasons.append("Soil moisture below optimal range")
            elif soil_moisture.status == 'wet':
                need_score -= 20
                reasons.append("Soil moisture above optimal range")
            
            # Rainfall factor
            expected_rainfall = rainfall_forecast.get('total_expected', 0)
            if expected_rainfall < daily_water_need * 0.5:  # Less than 50% of need
                need_score += 30
                reasons.append("Insufficient rainfall expected")
            elif expected_rainfall > daily_water_need * 1.5:  # More than 150% of need
                need_score -= 30
                reasons.append("Adequate rainfall expected")
            
            # Urgency factor
            if soil_moisture.days_until_critical and soil_moisture.days_until_critical <= 2:
                need_score += 30
                reasons.append("Critical moisture level approaching")
            
            # Determine urgency level
            if need_score >= 70:
                urgency = 'critical'
            elif need_score >= 50:
                urgency = 'high'
            elif need_score >= 30:
                urgency = 'medium'
            else:
                urgency = 'low'
            
            return {
                'score': need_score,
                'urgency': urgency,
                'reasons': reasons,
                'should_irrigate': need_score >= 30
            }
            
        except Exception as e:
            logger.error(f"Error determining irrigation need: {e}")
            return {'score': 50, 'urgency': 'medium', 'reasons': ['Error in calculation'], 'should_irrigate': True}
    
    def _generate_recommendation(self, irrigation_need: Dict[str, Any], 
                               soil_moisture: SoilMoistureStatus,
                               weather_forecast: Dict[str, Any],
                               daily_water_need: float,
                               area_hectares: float) -> IrrigationRecommendation:
        """Generate detailed irrigation recommendation"""
        try:
            if not irrigation_need['should_irrigate']:
                # No irrigation needed
                return IrrigationRecommendation(
                    scheduled_date=datetime.now() + timedelta(days=3),
                    water_amount_liters=0,
                    irrigation_method="none",
                    duration_minutes=0,
                    reasoning="No irrigation needed based on current conditions",
                    urgency_level="low",
                    weather_considerations=["Monitor soil moisture", "Check weather updates"]
                )
            
            # Calculate water amount
            moisture_deficit = max(0, soil_moisture.optimal_range[0] - soil_moisture.current_level)
            base_water_amount = daily_water_need * 1.5  # 1.5 days worth
            
            # Adjust for moisture deficit
            if moisture_deficit > 0:
                base_water_amount += moisture_deficit * area_hectares * 10000  # Convert to liters
            
            # Determine irrigation method
            if area_hectares < 2:
                irrigation_method = "drip"
                efficiency = 0.9
            elif area_hectares < 10:
                irrigation_method = "sprinkler"
                efficiency = 0.75
            else:
                irrigation_method = "flood"
                efficiency = 0.6
            
            # Adjust for efficiency
            water_amount = base_water_amount / efficiency
            
            # Calculate duration (assuming 100 L/min flow rate per hectare)
            flow_rate_per_hectare = 100  # L/min
            total_flow_rate = flow_rate_per_hectare * area_hectares
            duration_minutes = int(water_amount / total_flow_rate)
            
            # Determine timing
            rainfall_forecast = self._analyze_rainfall_forecast(weather_forecast)
            next_rain_day = rainfall_forecast.get('next_rain_day')
            
            if next_rain_day and next_rain_day <= 2:
                # Rain expected soon, delay irrigation
                scheduled_date = datetime.now() + timedelta(days=next_rain_day + 1)
                timing_reason = f"Delayed due to expected rain in {next_rain_day} days"
            elif irrigation_need['urgency'] == 'critical':
                # Immediate irrigation needed
                scheduled_date = datetime.now() + timedelta(hours=6)
                timing_reason = "Immediate irrigation required due to critical soil moisture"
            else:
                # Schedule for early morning
                scheduled_date = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(days=1)
                timing_reason = "Scheduled for early morning to minimize evaporation"
            
            # Weather considerations
            weather_considerations = []
            if rainfall_forecast['heavy_rain_days'] > 0:
                weather_considerations.append("Heavy rain expected - monitor for waterlogging")
            if weather_forecast['total_expected'] > 20:
                weather_considerations.append("Significant rainfall expected - may reduce irrigation need")
            
            # Generate reasoning
            reasoning_parts = [
                f"Soil moisture: {soil_moisture.status} ({soil_moisture.current_level:.2f})",
                f"Daily water need: {daily_water_need:.0f} L",
                f"Expected rainfall: {rainfall_forecast['total_expected']:.1f} mm",
                timing_reason
            ]
            reasoning = "; ".join(reasoning_parts)
            
            return IrrigationRecommendation(
                scheduled_date=scheduled_date,
                water_amount_liters=water_amount,
                irrigation_method=irrigation_method,
                duration_minutes=duration_minutes,
                reasoning=reasoning,
                urgency_level=irrigation_need['urgency'],
                weather_considerations=weather_considerations
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return IrrigationRecommendation(
                scheduled_date=datetime.now() + timedelta(days=1),
                water_amount_liters=daily_water_need,
                irrigation_method="sprinkler",
                duration_minutes=60,
                reasoning=f"Default recommendation due to error: {str(e)}",
                urgency_level="medium",
                weather_considerations=[]
            )
    
    async def _save_irrigation_schedule(self, farm_id: str, crop_type: str,
                                      recommendation: IrrigationRecommendation,
                                      session: Session):
        """Save irrigation schedule to database"""
        try:
            schedule = IrrigationSchedule(
                farm_id=farm_id,
                crop_type=crop_type,
                scheduled_date=recommendation.scheduled_date,
                water_amount_liters=recommendation.water_amount_liters,
                irrigation_method=recommendation.irrigation_method,
                weather_forecast={},  # Would include weather data
                soil_moisture_level=0.3,  # Would include actual measurement
                is_completed=False
            )
            
            session.add(schedule)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving irrigation schedule: {e}")
            session.rollback()
    
    async def get_weekly_irrigation_plan(self, farm_id: str, crop_type: str, 
                                       growth_stage: str, session: Session) -> List[IrrigationRecommendation]:
        """Generate a weekly irrigation plan"""
        try:
            weekly_plan = []
            
            for day in range(7):
                future_date = datetime.now() + timedelta(days=day)
                
                # Get recommendation for each day
                recommendation = await self.get_irrigation_recommendation(
                    farm_id, crop_type, growth_stage, session
                )
                
                # Adjust for the specific day
                recommendation.scheduled_date = future_date.replace(hour=6, minute=0)
                
                weekly_plan.append(recommendation)
            
            return weekly_plan
            
        except Exception as e:
            logger.error(f"Error generating weekly irrigation plan: {e}")
            return []
    
    def calculate_water_efficiency(self, irrigation_records: List[IrrigationSchedule]) -> Dict[str, float]:
        """Calculate irrigation efficiency metrics"""
        try:
            if not irrigation_records:
                return {}
            
            total_water_used = sum(record.water_amount_liters for record in irrigation_records)
            total_area = sum(record.farm.area_hectares for record in irrigation_records if record.farm)
            
            avg_water_per_hectare = total_water_used / total_area if total_area > 0 else 0
            
            # Calculate efficiency score (simplified)
            efficiency_score = min(100, max(0, 100 - (avg_water_per_hectare - 5000) / 100))
            
            return {
                'total_water_used_liters': total_water_used,
                'average_water_per_hectare': avg_water_per_hectare,
                'efficiency_score': efficiency_score,
                'irrigation_frequency': len(irrigation_records) / 30  # per month
            }
            
        except Exception as e:
            logger.error(f"Error calculating water efficiency: {e}")
            return {}