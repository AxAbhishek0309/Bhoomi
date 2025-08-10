"""
Market Forecaster Agent
Tracks crop prices and provides market intelligence using Agmarknet API
"""

import os
import requests
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from db.models import MarketPrice, Farm

logger = logging.getLogger(__name__)

@dataclass
class PriceForecast:
    crop_name: str
    market_name: str
    current_price: float
    predicted_price: float
    price_trend: str  # increasing, decreasing, stable
    confidence_level: float
    forecast_period_days: int
    factors_affecting_price: List[str]

@dataclass
class MarketInsight:
    crop_name: str
    best_markets: List[str]
    price_volatility: float
    seasonal_pattern: Dict[str, float]
    profit_potential: str  # high, medium, low
    risk_assessment: str

class MarketForecastAgent:
    """
    AI-powered market forecasting agent for crop price prediction and market intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.price_trend_days = config.get('price_trend_days', 30)
        self.supported_markets = config.get('supported_markets', [
            'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore'
        ])
        
        # Initialize web scraper (no API key needed)
        from utils.agmarknet_scraper import AgmarknetScraper
        self.agmarknet_scraper = AgmarknetScraper()
        
        # Crop-market mapping
        self.crop_market_mapping = {
            'wheat': ['delhi', 'punjab', 'haryana', 'uttar pradesh'],
            'rice': ['punjab', 'haryana', 'west bengal', 'andhra pradesh'],
            'tomato': ['maharashtra', 'karnataka', 'andhra pradesh'],
            'onion': ['maharashtra', 'karnataka', 'gujarat'],
            'potato': ['uttar pradesh', 'west bengal', 'bihar'],
            'cotton': ['gujarat', 'maharashtra', 'telangana'],
            'sugarcane': ['uttar pradesh', 'maharashtra', 'karnataka']
        }
    
    async def get_current_prices(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch current market prices for a specific crop - LIVE DATA ONLY
        
        Args:
            crop_name: Name of the crop
            markets: List of markets to check (optional)
            
        Returns:
            List of current price data
        """
        if markets is None:
            markets = self.crop_market_mapping.get(crop_name, self.supported_markets)
        
        current_prices = []
        
        # Try web scraping first (no API key needed)
        try:
            logger.info(f"Scraping Agmarknet for {crop_name} prices")
            scraped_prices = await self.agmarknet_scraper.get_crop_prices(crop_name, markets)
            if scraped_prices and len(scraped_prices) > 0:
                logger.info(f"Successfully scraped {len(scraped_prices)} price records from Agmarknet")
                current_prices.extend(scraped_prices)
            else:
                logger.warning("Agmarknet scraper returned no price data")
        except Exception as e:
            logger.error(f"Error with Agmarknet scraper: {e}")
        
        # Try alternative data sources if scraping fails
        if not current_prices:
            try:
                logger.info("Trying data.gov.in as fallback")
                api_prices = await self._fetch_from_data_gov_in(crop_name, markets)
                if api_prices:
                    current_prices.extend(api_prices)
            except Exception as e:
                logger.error(f"Error with data.gov.in: {e}")
        
        if not current_prices:
            raise Exception(f"All market price sources failed for {crop_name}. Please check:\n"
                           f"1. Network connection\n"
                           f"2. Crop name spelling: {crop_name}\n"
                           f"3. Agmarknet website accessibility\n"
                           f"Available markets: {markets}")
        
        return current_prices
    

    
    async def _fetch_from_data_gov_in(self, crop_name: str, markets: List[str]) -> List[Dict[str, Any]]:
        """Fetch prices from data.gov.in open data portal"""
        try:
            prices = []
            
            # data.gov.in has multiple agricultural datasets
            # Using the general commodity price dataset
            base_url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
            
            for market in markets:
                params = {
                    'api-key': self.api_key or 'YOUR_API_KEY',
                    'format': 'json',
                    'limit': 100,
                    'filters[commodity]': crop_name.lower(),
                    'filters[market]': market.lower()
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'records' in data:
                        for record in data['records']:
                            price_data = {
                                'crop_name': crop_name,
                                'market_name': record.get('market', market),
                                'state': record.get('state', ''),
                                'min_price': float(record.get('min_price', 0)),
                                'max_price': float(record.get('max_price', 0)),
                                'modal_price': float(record.get('modal_price', 0)),
                                'price_date': datetime.strptime(record.get('price_date', ''), '%Y-%m-%d') if record.get('price_date') else datetime.now(),
                                'unit': record.get('unit', 'quintal'),
                                'source': 'data_gov_in'
                            }
                            prices.append(price_data)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching from data.gov.in: {e}")
            return []
    
    async def _fetch_from_ncdex_api(self, crop_name: str, markets: List[str]) -> List[Dict[str, Any]]:
        """Fetch prices from NCDEX (National Commodity & Derivatives Exchange)"""
        try:
            prices = []
            
            # NCDEX provides commodity futures and spot prices
            # This is a simplified implementation - actual NCDEX API requires registration
            
            # Map crop names to NCDEX commodity codes
            ncdex_commodity_map = {
                'wheat': 'WHEAT',
                'rice': 'PADDY',
                'cotton': 'COTTON',
                'soybean': 'SOYBEAN',
                'corn': 'MAIZE',
                'turmeric': 'TURMERIC',
                'coriander': 'CORIANDER'
            }
            
            commodity_code = ncdex_commodity_map.get(crop_name.lower())
            if not commodity_code:
                return []
            
            # Simulate NCDEX data structure
            for market in markets:
                # In real implementation, you'd call NCDEX API here
                # For now, we'll return empty to force fallback to other APIs
                pass
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching from NCDEX: {e}")
            return []
    
    async def _fetch_from_csv_fallback(self, crop_name: str, markets: List[str]) -> List[Dict[str, Any]]:
        """Fetch prices from CSV fallback data"""
        try:
            if not os.path.exists(self.backup_csv):
                # Create sample fallback data
                await self._create_sample_price_data()
            
            df = pd.read_csv(self.backup_csv)
            
            # Filter for specific crop and markets
            filtered_df = df[
                (df['crop_name'].str.lower() == crop_name.lower()) &
                (df['market_name'].str.lower().isin([m.lower() for m in markets]))
            ]
            
            prices = []
            for _, row in filtered_df.iterrows():
                price_data = {
                    'crop_name': row['crop_name'],
                    'market_name': row['market_name'],
                    'state': row.get('state', ''),
                    'min_price': float(row['min_price']),
                    'max_price': float(row['max_price']),
                    'modal_price': float(row['modal_price']),
                    'price_date': pd.to_datetime(row['price_date']),
                    'unit': row.get('unit', 'quintal'),
                    'source': 'csv_fallback'
                }
                prices.append(price_data)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching from CSV fallback: {e}")
            return []
    
    def get_market_api_status(self) -> Dict[str, str]:
        """Get status of all market data sources"""
        status = {}
        
        # Check Agmarknet scraper
        try:
            if self.agmarknet_scraper.test_connection():
                status['agmarknet_scraper'] = 'Available - web scraping (no API key needed)'
            else:
                status['agmarknet_scraper'] = 'Offline - website not accessible'
        except:
            status['agmarknet_scraper'] = 'Error - scraper not initialized'
        
        # data.gov.in
        status['data_gov_in'] = 'Available - open government data portal'
        
        return status
    
    async def forecast_prices(self, crop_name: str, market_name: str, 
                            forecast_days: int = 30, session: Optional[Session] = None) -> PriceForecast:
        """
        Forecast crop prices using historical data and ML models
        
        Args:
            crop_name: Name of the crop
            market_name: Name of the market
            forecast_days: Number of days to forecast
            session: Database session (optional)
            
        Returns:
            PriceForecast with prediction details
        """
        try:
            # Get historical price data
            historical_data = await self._get_historical_prices(crop_name, market_name, session)
            
            if len(historical_data) < 10:
                # Not enough data for forecasting
                current_price = historical_data[-1]['modal_price'] if historical_data else 1500
                return PriceForecast(
                    crop_name=crop_name,
                    market_name=market_name,
                    current_price=current_price,
                    predicted_price=current_price,
                    price_trend='stable',
                    confidence_level=0.3,
                    forecast_period_days=forecast_days,
                    factors_affecting_price=['Insufficient historical data']
                )
            
            # Prepare data for ML model
            df = pd.DataFrame(historical_data)
            df['price_date'] = pd.to_datetime(df['price_date'])
            df = df.sort_values('price_date')
            
            # Feature engineering
            features = self._create_price_features(df)
            
            # Train prediction model
            model, scaler = self._train_price_model(features)
            
            # Make prediction
            predicted_price = self._predict_future_price(model, scaler, features, forecast_days)
            
            # Analyze trend
            current_price = df['modal_price'].iloc[-1]
            price_trend = self._analyze_price_trend(df['modal_price'].values)
            
            # Calculate confidence
            confidence_level = self._calculate_prediction_confidence(df, predicted_price)
            
            # Identify factors affecting price
            factors = self._identify_price_factors(df, crop_name)
            
            return PriceForecast(
                crop_name=crop_name,
                market_name=market_name,
                current_price=current_price,
                predicted_price=predicted_price,
                price_trend=price_trend,
                confidence_level=confidence_level,
                forecast_period_days=forecast_days,
                factors_affecting_price=factors
            )
            
        except Exception as e:
            logger.error(f"Error forecasting prices: {e}")
            return PriceForecast(
                crop_name=crop_name,
                market_name=market_name,
                current_price=1500,
                predicted_price=1500,
                price_trend='unknown',
                confidence_level=0.0,
                forecast_period_days=forecast_days,
                factors_affecting_price=[f'Forecasting error: {str(e)}']
            )
    
    async def _get_historical_prices(self, crop_name: str, market_name: str, 
                                   session: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Get historical price data from database or API"""
        try:
            historical_data = []
            
            # Try to get from database first
            if session:
                cutoff_date = datetime.now() - timedelta(days=90)
                
                prices = session.query(MarketPrice).filter(
                    MarketPrice.crop_name == crop_name,
                    MarketPrice.market_name == market_name,
                    MarketPrice.price_date >= cutoff_date
                ).order_by(MarketPrice.price_date).all()
                
                for price in prices:
                    historical_data.append({
                        'crop_name': price.crop_name,
                        'market_name': price.market_name,
                        'min_price': price.min_price,
                        'max_price': price.max_price,
                        'modal_price': price.modal_price,
                        'price_date': price.price_date
                    })
            
            # If not enough data, fetch from API/CSV
            if len(historical_data) < 30:
                api_data = await self._fetch_from_csv_fallback(crop_name, [market_name])
                historical_data.extend(api_data)
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return []
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for price prediction model"""
        try:
            features_df = df.copy()
            
            # Time-based features
            features_df['day_of_year'] = features_df['price_date'].dt.dayofyear
            features_df['month'] = features_df['price_date'].dt.month
            features_df['week_of_year'] = features_df['price_date'].dt.isocalendar().week
            
            # Price-based features
            features_df['price_ma_7'] = features_df['modal_price'].rolling(window=7).mean()
            features_df['price_ma_14'] = features_df['modal_price'].rolling(window=14).mean()
            features_df['price_volatility'] = features_df['modal_price'].rolling(window=7).std()
            features_df['price_change'] = features_df['modal_price'].pct_change()
            
            # Price range features
            features_df['price_range'] = features_df['max_price'] - features_df['min_price']
            features_df['price_position'] = (features_df['modal_price'] - features_df['min_price']) / features_df['price_range']
            
            # Lag features
            for lag in [1, 3, 7]:
                features_df[f'price_lag_{lag}'] = features_df['modal_price'].shift(lag)
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            return df
    
    def _train_price_model(self, features_df: pd.DataFrame) -> Tuple[LinearRegression, StandardScaler]:
        """Train ML model for price prediction"""
        try:
            # Select features for training
            feature_columns = [
                'day_of_year', 'month', 'week_of_year',
                'price_ma_7', 'price_ma_14', 'price_volatility',
                'price_range', 'price_position',
                'price_lag_1', 'price_lag_3', 'price_lag_7'
            ]
            
            # Prepare training data
            X = features_df[feature_columns].values
            y = features_df['modal_price'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error training price model: {e}")
            # Return dummy model
            return LinearRegression(), StandardScaler()
    
    def _predict_future_price(self, model: LinearRegression, scaler: StandardScaler,
                            features_df: pd.DataFrame, forecast_days: int) -> float:
        """Predict future price using trained model"""
        try:
            # Use the last row of features for prediction
            last_features = features_df.iloc[-1][[
                'day_of_year', 'month', 'week_of_year',
                'price_ma_7', 'price_ma_14', 'price_volatility',
                'price_range', 'price_position',
                'price_lag_1', 'price_lag_3', 'price_lag_7'
            ]].values.reshape(1, -1)
            
            # Scale features
            last_features_scaled = scaler.transform(last_features)
            
            # Make prediction
            predicted_price = model.predict(last_features_scaled)[0]
            
            # Adjust for forecast period (simple linear adjustment)
            trend_factor = 1 + (forecast_days / 365) * 0.05  # 5% annual trend
            predicted_price *= trend_factor
            
            return max(0, predicted_price)  # Ensure non-negative price
            
        except Exception as e:
            logger.error(f"Error predicting future price: {e}")
            return features_df['modal_price'].iloc[-1]  # Return last known price
    
    def _analyze_price_trend(self, prices: np.ndarray) -> str:
        """Analyze price trend from historical data"""
        try:
            if len(prices) < 5:
                return 'stable'
            
            # Calculate trend using linear regression
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Determine trend based on slope
            if slope > 50:  # Increasing by more than 50 per period
                return 'increasing'
            elif slope < -50:  # Decreasing by more than 50 per period
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error analyzing price trend: {e}")
            return 'unknown'
    
    def _calculate_prediction_confidence(self, df: pd.DataFrame, predicted_price: float) -> float:
        """Calculate confidence level for price prediction"""
        try:
            recent_prices = df['modal_price'].tail(10).values
            price_volatility = np.std(recent_prices)
            mean_price = np.mean(recent_prices)
            
            # Calculate confidence based on volatility
            volatility_ratio = price_volatility / mean_price
            
            if volatility_ratio < 0.1:
                confidence = 0.9
            elif volatility_ratio < 0.2:
                confidence = 0.7
            elif volatility_ratio < 0.3:
                confidence = 0.5
            else:
                confidence = 0.3
            
            # Adjust based on data availability
            data_points = len(df)
            if data_points < 20:
                confidence *= 0.7
            elif data_points < 50:
                confidence *= 0.85
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    def _identify_price_factors(self, df: pd.DataFrame, crop_name: str) -> List[str]:
        """Identify factors affecting crop prices"""
        factors = []
        
        try:
            # Seasonal factors
            current_month = datetime.now().month
            if crop_name in ['wheat', 'rice'] and current_month in [3, 4, 5]:
                factors.append("Harvest season - prices typically lower")
            elif crop_name in ['tomato', 'onion'] and current_month in [6, 7, 8]:
                factors.append("Monsoon season - supply disruptions possible")
            
            # Price volatility
            recent_volatility = df['modal_price'].tail(10).std()
            if recent_volatility > 200:
                factors.append("High price volatility observed")
            
            # Price trend
            recent_trend = self._analyze_price_trend(df['modal_price'].tail(15).values)
            if recent_trend == 'increasing':
                factors.append("Recent upward price trend")
            elif recent_trend == 'decreasing':
                factors.append("Recent downward price trend")
            
            # General factors
            factors.extend([
                "Weather conditions",
                "Government policies",
                "Export-import regulations",
                "Storage and transportation costs"
            ])
            
        except Exception as e:
            logger.error(f"Error identifying price factors: {e}")
        
        return factors[:5]  # Return top 5 factors
    
    async def get_market_insights(self, crop_name: str, session: Optional[Session] = None) -> MarketInsight:
        """Get comprehensive market insights for a crop"""
        try:
            # Get price data from multiple markets
            all_markets = self.crop_market_mapping.get(crop_name, self.supported_markets)
            market_data = {}
            
            for market in all_markets:
                prices = await self._get_historical_prices(crop_name, market, session)
                if prices:
                    market_data[market] = prices
            
            if not market_data:
                return MarketInsight(
                    crop_name=crop_name,
                    best_markets=[],
                    price_volatility=0.0,
                    seasonal_pattern={},
                    profit_potential='unknown',
                    risk_assessment='Data unavailable'
                )
            
            # Find best markets (highest average prices)
            market_averages = {}
            for market, prices in market_data.items():
                avg_price = np.mean([p['modal_price'] for p in prices])
                market_averages[market] = avg_price
            
            best_markets = sorted(market_averages.keys(), 
                                key=lambda x: market_averages[x], reverse=True)[:3]
            
            # Calculate overall volatility
            all_prices = []
            for prices in market_data.values():
                all_prices.extend([p['modal_price'] for p in prices])
            
            price_volatility = np.std(all_prices) if all_prices else 0.0
            
            # Analyze seasonal patterns
            seasonal_pattern = self._analyze_seasonal_pattern(market_data)
            
            # Assess profit potential
            profit_potential = self._assess_profit_potential(market_averages, price_volatility)
            
            # Risk assessment
            risk_assessment = self._assess_market_risk(price_volatility, len(all_prices))
            
            return MarketInsight(
                crop_name=crop_name,
                best_markets=best_markets,
                price_volatility=price_volatility,
                seasonal_pattern=seasonal_pattern,
                profit_potential=profit_potential,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error getting market insights: {e}")
            return MarketInsight(
                crop_name=crop_name,
                best_markets=[],
                price_volatility=0.0,
                seasonal_pattern={},
                profit_potential='unknown',
                risk_assessment=f'Analysis error: {str(e)}'
            )
    
    def _analyze_seasonal_pattern(self, market_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Analyze seasonal price patterns"""
        try:
            monthly_prices = {i: [] for i in range(1, 13)}
            
            for prices in market_data.values():
                for price_data in prices:
                    month = price_data['price_date'].month
                    monthly_prices[month].append(price_data['modal_price'])
            
            seasonal_pattern = {}
            for month, prices in monthly_prices.items():
                if prices:
                    seasonal_pattern[f'month_{month}'] = np.mean(prices)
            
            return seasonal_pattern
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal pattern: {e}")
            return {}
    
    def _assess_profit_potential(self, market_averages: Dict[str, float], volatility: float) -> str:
        """Assess profit potential based on prices and volatility"""
        try:
            if not market_averages:
                return 'unknown'
            
            avg_price = np.mean(list(market_averages.values()))
            
            # High price and low volatility = high profit potential
            if avg_price > 2000 and volatility < 200:
                return 'high'
            elif avg_price > 1500 and volatility < 300:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing profit potential: {e}")
            return 'unknown'
    
    def _assess_market_risk(self, volatility: float, data_points: int) -> str:
        """Assess market risk based on volatility and data availability"""
        try:
            risk_factors = []
            
            if volatility > 400:
                risk_factors.append("High price volatility")
            elif volatility > 200:
                risk_factors.append("Moderate price volatility")
            
            if data_points < 30:
                risk_factors.append("Limited historical data")
            
            if not risk_factors:
                return "Low risk - stable market conditions"
            else:
                return f"Risk factors: {', '.join(risk_factors)}"
                
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return "Risk assessment unavailable"
    
    async def save_price_data(self, price_data: List[Dict[str, Any]], session: Session):
        """Save price data to database"""
        try:
            for data in price_data:
                price_record = MarketPrice(
                    crop_name=data['crop_name'],
                    market_name=data['market_name'],
                    state=data.get('state', ''),
                    min_price=data['min_price'],
                    max_price=data['max_price'],
                    modal_price=data['modal_price'],
                    price_date=data['price_date'],
                    unit=data.get('unit', 'quintal'),
                    source=data.get('source', 'unknown')
                )
                session.add(price_record)
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving price data: {e}")
            session.rollback()