from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class Farm(Base):
    __tablename__ = 'farms'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    owner_name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    area_hectares = Column(Float, nullable=False)
    soil_type = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SoilReport(Base):
    __tablename__ = 'soil_reports'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    farm_id = Column(String(36), nullable=False)
    ph_level = Column(Float)
    nitrogen = Column(Float)  # percentage
    phosphorus = Column(Float)  # ppm
    potassium = Column(Float)  # ppm
    organic_matter = Column(Float)  # percentage
    moisture_content = Column(Float)  # percentage
    report_date = Column(DateTime, nullable=False)
    lab_name = Column(String(255))
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class CropRecommendation(Base):
    __tablename__ = 'crop_recommendations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    farm_id = Column(String(36), nullable=False)
    recommended_crops = Column(JSON, nullable=False)  # List of crop recommendations
    confidence_score = Column(Float)
    reasoning = Column(Text)
    weather_data = Column(JSON)
    market_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DiseaseDetection(Base):
    __tablename__ = 'disease_detections'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    farm_id = Column(String(36), nullable=False)
    image_path = Column(String(500), nullable=False)
    crop_type = Column(String(100), nullable=False)
    detected_disease = Column(String(255))
    confidence_score = Column(Float)
    severity_level = Column(String(50))  # low, medium, high
    treatment_recommendations = Column(JSON)
    detection_date = Column(DateTime, default=datetime.utcnow)

class IrrigationSchedule(Base):
    __tablename__ = 'irrigation_schedules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id = Column(UUID(as_uuid=True), nullable=False)
    crop_type = Column(String(100), nullable=False)
    scheduled_date = Column(DateTime, nullable=False)
    water_amount_liters = Column(Float)
    irrigation_method = Column(String(100))  # drip, sprinkler, flood
    weather_forecast = Column(JSON)
    soil_moisture_level = Column(Float)
    is_completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketPrice(Base):
    __tablename__ = 'market_prices'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    crop_name = Column(String(100), nullable=False)
    market_name = Column(String(255), nullable=False)
    state = Column(String(100), nullable=False)
    min_price = Column(Float)
    max_price = Column(Float)
    modal_price = Column(Float)
    price_date = Column(DateTime, nullable=False)
    unit = Column(String(50), default='quintal')
    source = Column(String(100))  # agmarknet, manual, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

class YieldPrediction(Base):
    __tablename__ = 'yield_predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id = Column(UUID(as_uuid=True), nullable=False)
    crop_type = Column(String(100), nullable=False)
    predicted_yield_tons_per_hectare = Column(Float)
    confidence_interval = Column(JSON)  # {lower: float, upper: float}
    prediction_date = Column(DateTime, default=datetime.utcnow)
    harvest_date = Column(DateTime)
    input_features = Column(JSON)  # NDVI, weather, soil data used
    model_version = Column(String(50))

class WeatherData(Base):
    __tablename__ = 'weather_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    temperature_celsius = Column(Float)
    humidity_percent = Column(Float)
    rainfall_mm = Column(Float)
    wind_speed_kmh = Column(Float)
    pressure_hpa = Column(Float)
    uv_index = Column(Float)
    recorded_at = Column(DateTime, nullable=False)
    source = Column(String(100))  # openweather, nasa_power, etc.
    forecast_data = Column(JSON)  # 7-day forecast

class VoiceQuery(Base):
    __tablename__ = 'voice_queries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    farm_id = Column(UUID(as_uuid=True))
    audio_file_path = Column(String(500))
    transcribed_text = Column(Text)
    detected_language = Column(String(10))
    confidence_score = Column(Float)
    agent_response = Column(Text)
    processing_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)