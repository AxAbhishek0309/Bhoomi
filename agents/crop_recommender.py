"""
Crop Recommender Agent
Uses LLM and environmental data to recommend optimal crops for a farm
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sqlalchemy.orm import Session

from db.models import Farm, SoilReport, WeatherData, MarketPrice, CropRecommendation
from utils.weather_fetcher import WeatherFetcher
from utils.file_parser import parse_soil_report

logger = logging.getLogger(__name__)

@dataclass
class CropSuitability:
    crop_name: str
    suitability_score: float  # 0-100
    reasoning: str
    expected_yield: Optional[float] = None
    market_potential: Optional[str] = None
    risk_factors: List[str] = None

class CropRecommenderAgent:
    """
    Intelligent crop recommendation agent using LLM and multi-factor analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = OpenAI(
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 1000)
        )
        self.weather_fetcher = WeatherFetcher()
        
        # Create the recommendation prompt template
        self.prompt_template = PromptTemplate(
            input_variables=[
                "soil_data", "weather_data", "market_data", 
                "farm_location", "farm_size", "season"
            ],
            template=self._get_prompt_template()
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _get_prompt_template(self) -> str:
        """Get the LLM prompt template for crop recommendations"""
        return """
        You are an expert agricultural advisor with deep knowledge of crop science, soil management, 
        weather patterns, and market economics. Analyze the following farm data and provide 
        crop recommendations.

        FARM INFORMATION:
        Location: {farm_location}
        Farm Size: {farm_size} hectares
        Current Season: {season}

        SOIL DATA:
        {soil_data}

        WEATHER DATA:
        {weather_data}

        MARKET DATA:
        {market_data}

        INSTRUCTIONS:
        1. Recommend the top 3-5 most suitable crops for this farm
        2. For each crop, provide:
           - Suitability score (0-100)
           - Detailed reasoning based on soil, weather, and market factors
           - Expected yield estimate
           - Market potential assessment
           - Key risk factors to consider
        3. Consider crop rotation benefits and soil health
        4. Factor in water requirements and irrigation needs
        5. Include both food crops and cash crops where appropriate

        Provide your response in the following JSON format:
        {{
            "recommendations": [
                {{
                    "crop_name": "crop_name",
                    "suitability_score": 85,
                    "reasoning": "detailed explanation",
                    "expected_yield": "tons per hectare",
                    "market_potential": "high/medium/low with explanation",
                    "risk_factors": ["factor1", "factor2"],
                    "planting_season": "optimal planting time",
                    "water_requirements": "irrigation needs"
                }}
            ],
            "general_advice": "overall farming recommendations",
            "soil_improvements": "suggested soil amendments"
        }}
        """
    
    async def recommend_crops(self, farm_id: str, session: Session) -> Dict[str, Any]:
        """
        Generate crop recommendations for a specific farm
        
        Args:
            farm_id: UUID of the farm
            session: Database session
            
        Returns:
            Dictionary containing crop recommendations and analysis
        """
        try:
            # Get farm data
            farm = session.query(Farm).filter(Farm.id == farm_id).first()
            if not farm:
                raise ValueError(f"Farm with ID {farm_id} not found")
            
            # Get latest soil report
            soil_report = session.query(SoilReport).filter(
                SoilReport.farm_id == farm_id
            ).order_by(SoilReport.report_date.desc()).first()
            
            # Get weather data
            weather_data = await self._get_weather_data(farm)
            
            # Get market data
            market_data = self._get_market_data(session)
            
            # Prepare input data for LLM
            input_data = {
                "farm_location": f"Lat: {farm.location.y}, Lon: {farm.location.x}",
                "farm_size": farm.area_hectares,
                "season": self._get_current_season(),
                "soil_data": self._format_soil_data(soil_report),
                "weather_data": self._format_weather_data(weather_data),
                "market_data": self._format_market_data(market_data)
            }
            
            # Generate recommendations using LLM
            response = await self.chain.arun(**input_data)
            
            # Parse LLM response
            recommendations = self._parse_llm_response(response)
            
            # Save recommendations to database
            await self._save_recommendations(farm_id, recommendations, session)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating crop recommendations: {e}")
            return {"error": str(e), "recommendations": []}
    
    async def _get_weather_data(self, farm: Farm) -> Dict[str, Any]:
        """Get weather data for the farm location"""
        try:
            latitude = farm.location.y
            longitude = farm.location.x
            
            current_weather = await self.weather_fetcher.get_current_weather(latitude, longitude)
            forecast = await self.weather_fetcher.get_forecast(latitude, longitude, days=7)
            
            return {
                "current": current_weather,
                "forecast": forecast
            }
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {}
    
    def _get_market_data(self, session: Session) -> List[Dict[str, Any]]:
        """Get recent market price data"""
        try:
            # Get prices from last 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            prices = session.query(MarketPrice).filter(
                MarketPrice.price_date >= cutoff_date
            ).order_by(MarketPrice.price_date.desc()).limit(50).all()
            
            market_data = []
            for price in prices:
                market_data.append({
                    "crop": price.crop_name,
                    "market": price.market_name,
                    "modal_price": price.modal_price,
                    "date": price.price_date.isoformat()
                })
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return []
    
    def _get_current_season(self) -> str:
        """Determine current agricultural season"""
        month = datetime.now().month
        
        if month in [6, 7, 8, 9]:  # June to September
            return "Kharif (Monsoon)"
        elif month in [10, 11, 12, 1, 2, 3]:  # October to March
            return "Rabi (Winter)"
        else:  # April, May
            return "Zaid (Summer)"
    
    def _format_soil_data(self, soil_report: Optional[SoilReport]) -> str:
        """Format soil data for LLM input"""
        if not soil_report:
            return "No recent soil report available"
        
        return f"""
        pH Level: {soil_report.ph_level}
        Nitrogen: {soil_report.nitrogen}%
        Phosphorus: {soil_report.phosphorus} ppm
        Potassium: {soil_report.potassium} ppm
        Organic Matter: {soil_report.organic_matter}%
        Moisture Content: {soil_report.moisture_content}%
        Report Date: {soil_report.report_date.strftime('%Y-%m-%d')}
        """
    
    def _format_weather_data(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data for LLM input"""
        if not weather_data:
            return "Weather data unavailable"
        
        current = weather_data.get('current', {})
        forecast = weather_data.get('forecast', {})
        
        return f"""
        Current Temperature: {current.get('temperature', 'N/A')}°C
        Humidity: {current.get('humidity', 'N/A')}%
        Recent Rainfall: {current.get('rainfall', 'N/A')} mm
        7-day Forecast: {json.dumps(forecast, indent=2)}
        """
    
    def _format_market_data(self, market_data: List[Dict[str, Any]]) -> str:
        """Format market data for LLM input"""
        if not market_data:
            return "Market data unavailable"
        
        # Group by crop and calculate averages
        crop_prices = {}
        for item in market_data:
            crop = item['crop']
            if crop not in crop_prices:
                crop_prices[crop] = []
            crop_prices[crop].append(item['modal_price'])
        
        formatted_data = "Recent Market Prices (₹/quintal):\n"
        for crop, prices in crop_prices.items():
            avg_price = sum(prices) / len(prices)
            formatted_data += f"- {crop.title()}: ₹{avg_price:.0f} (avg)\n"
        
        return formatted_data
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: create structured response from text
                return {
                    "recommendations": [],
                    "general_advice": response,
                    "error": "Could not parse structured response"
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "recommendations": [],
                "general_advice": response,
                "error": f"JSON parsing error: {str(e)}"
            }
    
    async def _save_recommendations(self, farm_id: str, recommendations: Dict[str, Any], 
                                  session: Session):
        """Save recommendations to database"""
        try:
            recommendation_record = CropRecommendation(
                farm_id=farm_id,
                recommended_crops=recommendations.get('recommendations', []),
                confidence_score=self._calculate_confidence_score(recommendations),
                reasoning=recommendations.get('general_advice', ''),
                weather_data={},  # Would include weather data used
                market_data={}    # Would include market data used
            )
            
            session.add(recommendation_record)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            session.rollback()
    
    def _calculate_confidence_score(self, recommendations: Dict[str, Any]) -> float:
        """Calculate overall confidence score for recommendations"""
        recs = recommendations.get('recommendations', [])
        if not recs:
            return 0.0
        
        # Average of individual suitability scores
        scores = [rec.get('suitability_score', 0) for rec in recs]
        return sum(scores) / len(scores) if scores else 0.0