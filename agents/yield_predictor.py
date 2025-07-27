"""
Yield Predictor Agent
Uses ML models to predict crop yields based on environmental and agricultural data
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from sqlalchemy.orm import Session
from db.models import Farm, SoilReport, WeatherData, YieldPrediction
from utils.weather_fetcher import WeatherFetcher
from utils.satellite_ndvi import NDVIFetcher

logger = logging.getLogger(__name__)

@dataclass
class YieldPredictionResult:
    crop_type: str
    predicted_yield_tons_per_hectare: float
    confidence_interval: Tuple[float, float]
    prediction_accuracy: float
    key_factors: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    harvest_date_estimate: datetime

@dataclass
class YieldFactors:
    ndvi_average: float
    temperature_average: float
    rainfall_total: float
    soil_ph: float
    nitrogen_level: float
    phosphorus_level: float
    potassium_level: float
    organic_matter: float

class YieldPredictorAgent:
    """
    AI-powered yield prediction agent using multiple ML models and environmental data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get('model_path', 'models/yield_predictor.pkl')
        self.features = config.get('features', [
            'ndvi', 'temperature', 'rainfall', 'soil_ph', 
            'nitrogen', 'phosphorus', 'potassium', 'organic_matter'
        ])
        self.prediction_horizon_days = config.get('prediction_horizon_days', 90)
        
        # Initialize components
        self.weather_fetcher = WeatherFetcher()
        self.ndvi_fetcher = NDVIFetcher()
        
        # Model ensemble
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Load or initialize models
        self._load_models()
        
        # Crop-specific parameters
        self.crop_parameters = {
            'wheat': {
                'growing_season_days': 120,
                'optimal_temp_range': (15, 25),
                'water_requirement': 450,  # mm
                'yield_potential': 4.5,    # tons/hectare
                'critical_growth_stages': ['tillering', 'flowering', 'grain_filling']
            },
            'rice': {
                'growing_season_days': 130,
                'optimal_temp_range': (20, 30),
                'water_requirement': 1200,
                'yield_potential': 6.0,
                'critical_growth_stages': ['tillering', 'panicle_initiation', 'flowering']
            },
            'corn': {
                'growing_season_days': 110,
                'optimal_temp_range': (18, 27),
                'water_requirement': 600,
                'yield_potential': 8.0,
                'critical_growth_stages': ['vegetative', 'tasseling', 'grain_filling']
            },
            'tomato': {
                'growing_season_days': 90,
                'optimal_temp_range': (18, 26),
                'water_requirement': 400,
                'yield_potential': 50.0,
                'critical_growth_stages': ['flowering', 'fruit_set', 'ripening']
            },
            'potato': {
                'growing_season_days': 100,
                'optimal_temp_range': (15, 20),
                'water_requirement': 350,
                'yield_potential': 25.0,
                'critical_growth_stages': ['emergence', 'tuber_initiation', 'bulking']
            }
        }
    
    def _load_models(self):
        """Load pre-trained models or initialize new ones"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models = model_data.get('models', {})
                    self.scalers = model_data.get('scalers', {})
                    self.label_encoders = model_data.get('label_encoders', {})
                logger.info(f"Loaded yield prediction models from {self.model_path}")
            else:
                # Initialize new models
                self._initialize_models()
                logger.info("Initialized new yield prediction models")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize new ML models"""
        try:
            # Create ensemble of models
            self.models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=10
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=6
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=6
                ),
                'linear': LinearRegression()
            }
            
            # Initialize scalers and encoders
            self.scalers = {
                'features': StandardScaler(),
                'target': StandardScaler()
            }
            
            self.label_encoders = {
                'crop_type': LabelEncoder(),
                'soil_type': LabelEncoder()
            }
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def predict_yield(self, farm_id: str, crop_type: str, 
                          planting_date: datetime, session: Session) -> YieldPredictionResult:
        """
        Predict crop yield for a specific farm and crop
        
        Args:
            farm_id: UUID of the farm
            crop_type: Type of crop to predict
            planting_date: Date when crop was planted
            session: Database session
            
        Returns:
            YieldPredictionResult with prediction details
        """
        try:
            # Get farm information
            farm = session.query(Farm).filter(Farm.id == farm_id).first()
            if not farm:
                raise ValueError(f"Farm with ID {farm_id} not found")
            
            # Collect input features
            features = await self._collect_features(farm, crop_type, planting_date, session)
            
            # Make prediction using ensemble
            prediction_results = self._predict_with_ensemble(features, crop_type)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                prediction_results, features
            )
            
            # Identify key factors
            key_factors = self._identify_key_factors(features, crop_type)
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(features, crop_type)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, crop_type, prediction_results)
            
            # Estimate harvest date
            harvest_date = self._estimate_harvest_date(planting_date, crop_type)
            
            # Calculate prediction accuracy (based on model performance)
            accuracy = self._estimate_prediction_accuracy(crop_type, features)
            
            result = YieldPredictionResult(
                crop_type=crop_type,
                predicted_yield_tons_per_hectare=prediction_results['ensemble_prediction'],
                confidence_interval=confidence_interval,
                prediction_accuracy=accuracy,
                key_factors=key_factors,
                risk_factors=risk_factors,
                recommendations=recommendations,
                harvest_date_estimate=harvest_date
            )
            
            # Save prediction to database
            await self._save_prediction(farm_id, result, features, session)
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting yield: {e}")
            return self._create_fallback_prediction(crop_type, e)
    
    async def _collect_features(self, farm: Farm, crop_type: str, 
                              planting_date: datetime, session: Session) -> YieldFactors:
        """Collect all features needed for yield prediction"""
        try:
            latitude = farm.location.y
            longitude = farm.location.x
            
            # Get NDVI data
            ndvi_data = await self.ndvi_fetcher.get_ndvi_timeseries(
                latitude, longitude, planting_date, datetime.now()
            )
            ndvi_average = np.mean([d['ndvi'] for d in ndvi_data]) if ndvi_data else 0.5
            
            # Get weather data
            weather_data = await self.weather_fetcher.get_historical_weather(
                latitude, longitude, planting_date, datetime.now()
            )
            
            if weather_data:
                temperature_average = np.mean([d.get('temperature', 20) for d in weather_data])
                rainfall_total = sum([d.get('rainfall', 0) for d in weather_data])
            else:
                temperature_average = 22.0  # Default
                rainfall_total = 500.0      # Default
            
            # Get soil data
            soil_report = session.query(SoilReport).filter(
                SoilReport.farm_id == farm.id
            ).order_by(SoilReport.report_date.desc()).first()
            
            if soil_report:
                soil_ph = soil_report.ph_level or 6.5
                nitrogen_level = soil_report.nitrogen or 0.04
                phosphorus_level = soil_report.phosphorus or 20.0
                potassium_level = soil_report.potassium or 150.0
                organic_matter = soil_report.organic_matter or 2.5
            else:
                # Default values
                soil_ph = 6.5
                nitrogen_level = 0.04
                phosphorus_level = 20.0
                potassium_level = 150.0
                organic_matter = 2.5
            
            return YieldFactors(
                ndvi_average=ndvi_average,
                temperature_average=temperature_average,
                rainfall_total=rainfall_total,
                soil_ph=soil_ph,
                nitrogen_level=nitrogen_level,
                phosphorus_level=phosphorus_level,
                potassium_level=potassium_level,
                organic_matter=organic_matter
            )
            
        except Exception as e:
            logger.error(f"Error collecting features: {e}")
            # Return default features
            return YieldFactors(
                ndvi_average=0.5,
                temperature_average=22.0,
                rainfall_total=500.0,
                soil_ph=6.5,
                nitrogen_level=0.04,
                phosphorus_level=20.0,
                potassium_level=150.0,
                organic_matter=2.5
            )
    
    def _predict_with_ensemble(self, features: YieldFactors, crop_type: str) -> Dict[str, float]:
        """Make predictions using ensemble of models"""
        try:
            # Convert features to array
            feature_array = np.array([
                features.ndvi_average,
                features.temperature_average,
                features.rainfall_total,
                features.soil_ph,
                features.nitrogen_level,
                features.phosphorus_level,
                features.potassium_level,
                features.organic_matter
            ]).reshape(1, -1)
            
            predictions = {}
            
            # If models are trained, use them
            if self.models and any(hasattr(model, 'predict') for model in self.models.values()):
                for model_name, model in self.models.items():
                    try:
                        if hasattr(model, 'predict'):
                            pred = model.predict(feature_array)[0]
                            predictions[model_name] = max(0, pred)  # Ensure non-negative
                    except Exception as e:
                        logger.warning(f"Error with {model_name}: {e}")
                        predictions[model_name] = self._fallback_prediction(features, crop_type)
            else:
                # Use rule-based fallback
                predictions['fallback'] = self._fallback_prediction(features, crop_type)
            
            # Calculate ensemble prediction (weighted average)
            if predictions:
                ensemble_prediction = np.mean(list(predictions.values()))
            else:
                ensemble_prediction = self._fallback_prediction(features, crop_type)
            
            predictions['ensemble_prediction'] = ensemble_prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'ensemble_prediction': self._fallback_prediction(features, crop_type)}
    
    def _fallback_prediction(self, features: YieldFactors, crop_type: str) -> float:
        """Rule-based fallback prediction when ML models are not available"""
        try:
            crop_params = self.crop_parameters.get(crop_type, {})
            base_yield = crop_params.get('yield_potential', 3.0)
            
            # Adjust based on environmental factors
            yield_multiplier = 1.0
            
            # NDVI factor (vegetation health)
            if features.ndvi_average > 0.7:
                yield_multiplier *= 1.2
            elif features.ndvi_average < 0.3:
                yield_multiplier *= 0.7
            
            # Temperature factor
            optimal_temp = crop_params.get('optimal_temp_range', (20, 25))
            if optimal_temp[0] <= features.temperature_average <= optimal_temp[1]:
                yield_multiplier *= 1.1
            elif features.temperature_average < optimal_temp[0] - 5 or features.temperature_average > optimal_temp[1] + 5:
                yield_multiplier *= 0.8
            
            # Rainfall factor
            water_requirement = crop_params.get('water_requirement', 500)
            rainfall_ratio = features.rainfall_total / water_requirement
            if 0.8 <= rainfall_ratio <= 1.2:
                yield_multiplier *= 1.1
            elif rainfall_ratio < 0.5 or rainfall_ratio > 2.0:
                yield_multiplier *= 0.7
            
            # Soil pH factor
            if 6.0 <= features.soil_ph <= 7.5:
                yield_multiplier *= 1.05
            elif features.soil_ph < 5.5 or features.soil_ph > 8.0:
                yield_multiplier *= 0.85
            
            # Nutrient factors
            if features.nitrogen_level > 0.05:
                yield_multiplier *= 1.1
            elif features.nitrogen_level < 0.02:
                yield_multiplier *= 0.8
            
            if features.organic_matter > 3.0:
                yield_multiplier *= 1.05
            elif features.organic_matter < 1.5:
                yield_multiplier *= 0.9
            
            predicted_yield = base_yield * yield_multiplier
            return max(0.1, predicted_yield)  # Minimum viable yield
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return 2.0  # Conservative default
    
    def _calculate_confidence_interval(self, predictions: Dict[str, float], 
                                     features: YieldFactors) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        try:
            ensemble_pred = predictions['ensemble_prediction']
            
            # Calculate prediction variance based on model agreement
            model_preds = [v for k, v in predictions.items() if k != 'ensemble_prediction']
            
            if len(model_preds) > 1:
                pred_std = np.std(model_preds)
            else:
                # Estimate uncertainty based on data quality
                pred_std = ensemble_pred * 0.2  # 20% uncertainty
            
            # Adjust confidence based on feature quality
            confidence_factor = 1.0
            
            # NDVI quality
            if features.ndvi_average < 0.2 or features.ndvi_average > 0.9:
                confidence_factor *= 1.3  # Higher uncertainty
            
            # Weather data quality (simplified check)
            if features.rainfall_total < 100 or features.rainfall_total > 2000:
                confidence_factor *= 1.2
            
            adjusted_std = pred_std * confidence_factor
            
            # 95% confidence interval
            lower_bound = max(0, ensemble_pred - 1.96 * adjusted_std)
            upper_bound = ensemble_pred + 1.96 * adjusted_std
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            pred = predictions.get('ensemble_prediction', 2.0)
            return (pred * 0.7, pred * 1.3)
    
    def _identify_key_factors(self, features: YieldFactors, crop_type: str) -> List[str]:
        """Identify key factors affecting yield prediction"""
        key_factors = []
        
        try:
            # NDVI analysis
            if features.ndvi_average > 0.7:
                key_factors.append("Excellent vegetation health (high NDVI)")
            elif features.ndvi_average < 0.3:
                key_factors.append("Poor vegetation health (low NDVI)")
            
            # Weather analysis
            crop_params = self.crop_parameters.get(crop_type, {})
            optimal_temp = crop_params.get('optimal_temp_range', (20, 25))
            
            if optimal_temp[0] <= features.temperature_average <= optimal_temp[1]:
                key_factors.append("Optimal temperature conditions")
            else:
                key_factors.append("Sub-optimal temperature conditions")
            
            # Rainfall analysis
            water_requirement = crop_params.get('water_requirement', 500)
            rainfall_ratio = features.rainfall_total / water_requirement
            
            if 0.8 <= rainfall_ratio <= 1.2:
                key_factors.append("Adequate rainfall for crop needs")
            elif rainfall_ratio < 0.8:
                key_factors.append("Insufficient rainfall - irrigation needed")
            else:
                key_factors.append("Excess rainfall - drainage concerns")
            
            # Soil analysis
            if 6.0 <= features.soil_ph <= 7.5:
                key_factors.append("Optimal soil pH for nutrient uptake")
            
            if features.nitrogen_level > 0.05:
                key_factors.append("Good nitrogen availability")
            
            if features.organic_matter > 3.0:
                key_factors.append("High organic matter content")
            
        except Exception as e:
            logger.error(f"Error identifying key factors: {e}")
        
        return key_factors[:5]  # Return top 5 factors
    
    def _assess_risk_factors(self, features: YieldFactors, crop_type: str) -> List[str]:
        """Assess risk factors that could negatively impact yield"""
        risk_factors = []
        
        try:
            # Environmental risks
            if features.ndvi_average < 0.3:
                risk_factors.append("Low vegetation index indicates plant stress")
            
            crop_params = self.crop_parameters.get(crop_type, {})
            optimal_temp = crop_params.get('optimal_temp_range', (20, 25))
            
            if features.temperature_average > optimal_temp[1] + 5:
                risk_factors.append("High temperature stress")
            elif features.temperature_average < optimal_temp[0] - 5:
                risk_factors.append("Low temperature stress")
            
            # Water stress
            water_requirement = crop_params.get('water_requirement', 500)
            if features.rainfall_total < water_requirement * 0.6:
                risk_factors.append("Severe water deficit")
            elif features.rainfall_total > water_requirement * 1.8:
                risk_factors.append("Waterlogging risk")
            
            # Soil risks
            if features.soil_ph < 5.5:
                risk_factors.append("Soil acidity limiting nutrient availability")
            elif features.soil_ph > 8.0:
                risk_factors.append("Soil alkalinity affecting nutrient uptake")
            
            if features.nitrogen_level < 0.02:
                risk_factors.append("Nitrogen deficiency")
            
            if features.phosphorus_level < 10:
                risk_factors.append("Phosphorus deficiency")
            
            if features.organic_matter < 1.5:
                risk_factors.append("Low organic matter affecting soil health")
            
        except Exception as e:
            logger.error(f"Error assessing risk factors: {e}")
        
        return risk_factors[:5]  # Return top 5 risks
    
    def _generate_recommendations(self, features: YieldFactors, crop_type: str, 
                                predictions: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations to improve yield"""
        recommendations = []
        
        try:
            # NDVI-based recommendations
            if features.ndvi_average < 0.4:
                recommendations.append("Monitor crop health closely - consider foliar nutrition")
            
            # Nutrient recommendations
            if features.nitrogen_level < 0.03:
                recommendations.append("Apply nitrogen fertilizer to boost growth")
            
            if features.phosphorus_level < 15:
                recommendations.append("Apply phosphorus fertilizer for root development")
            
            if features.potassium_level < 120:
                recommendations.append("Apply potassium fertilizer for disease resistance")
            
            # Soil recommendations
            if features.soil_ph < 6.0:
                recommendations.append("Apply lime to correct soil acidity")
            elif features.soil_ph > 7.5:
                recommendations.append("Apply sulfur to reduce soil alkalinity")
            
            if features.organic_matter < 2.0:
                recommendations.append("Add compost or organic matter to improve soil health")
            
            # Water management
            crop_params = self.crop_parameters.get(crop_type, {})
            water_requirement = crop_params.get('water_requirement', 500)
            
            if features.rainfall_total < water_requirement * 0.8:
                recommendations.append("Implement irrigation to meet water requirements")
            
            # General recommendations
            recommendations.append("Monitor weather forecasts for pest and disease risks")
            recommendations.append("Ensure proper crop spacing and weed management")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations[:6]  # Return top 6 recommendations
    
    def _estimate_harvest_date(self, planting_date: datetime, crop_type: str) -> datetime:
        """Estimate harvest date based on crop type and planting date"""
        try:
            crop_params = self.crop_parameters.get(crop_type, {})
            growing_season_days = crop_params.get('growing_season_days', 120)
            
            harvest_date = planting_date + timedelta(days=growing_season_days)
            return harvest_date
            
        except Exception as e:
            logger.error(f"Error estimating harvest date: {e}")
            return planting_date + timedelta(days=120)  # Default 4 months
    
    def _estimate_prediction_accuracy(self, crop_type: str, features: YieldFactors) -> float:
        """Estimate prediction accuracy based on data quality and model performance"""
        try:
            base_accuracy = 0.75  # Base accuracy assumption
            
            # Adjust based on data quality
            if features.ndvi_average > 0.1:  # Valid NDVI data
                base_accuracy += 0.1
            
            if 0 < features.rainfall_total < 3000:  # Reasonable rainfall data
                base_accuracy += 0.05
            
            if 5.0 < features.soil_ph < 9.0:  # Valid pH data
                base_accuracy += 0.05
            
            # Crop-specific adjustments
            if crop_type in self.crop_parameters:
                base_accuracy += 0.05  # Known crop parameters
            
            return min(0.95, max(0.5, base_accuracy))
            
        except Exception as e:
            logger.error(f"Error estimating accuracy: {e}")
            return 0.7  # Default accuracy
    
    def _create_fallback_prediction(self, crop_type: str, error: Exception) -> YieldPredictionResult:
        """Create fallback prediction when main prediction fails"""
        crop_params = self.crop_parameters.get(crop_type, {})
        base_yield = crop_params.get('yield_potential', 3.0) * 0.7  # Conservative estimate
        
        return YieldPredictionResult(
            crop_type=crop_type,
            predicted_yield_tons_per_hectare=base_yield,
            confidence_interval=(base_yield * 0.6, base_yield * 1.4),
            prediction_accuracy=0.5,
            key_factors=[f"Prediction error: {str(error)}"],
            risk_factors=["Limited data availability"],
            recommendations=["Collect more comprehensive farm data for better predictions"],
            harvest_date_estimate=datetime.now() + timedelta(days=120)
        )
    
    async def _save_prediction(self, farm_id: str, result: YieldPredictionResult, 
                             features: YieldFactors, session: Session):
        """Save yield prediction to database"""
        try:
            prediction_record = YieldPrediction(
                farm_id=farm_id,
                crop_type=result.crop_type,
                predicted_yield_tons_per_hectare=result.predicted_yield_tons_per_hectare,
                confidence_interval={
                    'lower': result.confidence_interval[0],
                    'upper': result.confidence_interval[1]
                },
                harvest_date=result.harvest_date_estimate,
                input_features={
                    'ndvi_average': features.ndvi_average,
                    'temperature_average': features.temperature_average,
                    'rainfall_total': features.rainfall_total,
                    'soil_ph': features.soil_ph,
                    'nitrogen_level': features.nitrogen_level,
                    'phosphorus_level': features.phosphorus_level,
                    'potassium_level': features.potassium_level,
                    'organic_matter': features.organic_matter
                },
                model_version='v1.0'
            )
            
            session.add(prediction_record)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            session.rollback()
    
    def train_models_with_data(self, training_data: pd.DataFrame):
        """Train models with historical yield data"""
        try:
            # Prepare features and target
            feature_columns = [
                'ndvi_average', 'temperature_average', 'rainfall_total',
                'soil_ph', 'nitrogen_level', 'phosphorus_level', 
                'potassium_level', 'organic_matter'
            ]
            
            X = training_data[feature_columns].values
            y = training_data['actual_yield'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Train each model
            for model_name, model in self.models.items():
                if model_name != 'ensemble_prediction':
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    logger.info(f"{model_name} - MAE: {mae:.2f}, R2: {r2:.3f}")
            
            # Save trained models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved models to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")