-- AgriBotX Pro PostgreSQL Schema with PostGIS
-- This script initializes the database with spatial extensions

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create custom types
CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE irrigation_method AS ENUM ('drip', 'sprinkler', 'flood', 'micro');
CREATE TYPE data_quality AS ENUM ('good', 'fair', 'poor', 'unknown');

-- Create farms table with spatial data
CREATE TABLE IF NOT EXISTS farms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    owner_name VARCHAR(255) NOT NULL,
    location GEOMETRY(POINT, 4326) NOT NULL,
    area_hectares DECIMAL(10,2) NOT NULL,
    soil_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial index on farms location
CREATE INDEX IF NOT EXISTS idx_farms_location ON farms USING GIST (location);

-- Create soil_reports table
CREATE TABLE IF NOT EXISTS soil_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id),
    ph_level DECIMAL(4,2),
    nitrogen DECIMAL(6,4),
    phosphorus DECIMAL(8,2),
    potassium DECIMAL(8,2),
    organic_matter DECIMAL(6,2),
    moisture_content DECIMAL(6,2),
    report_date TIMESTAMP NOT NULL,
    lab_name VARCHAR(255),
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on soil_reports
CREATE INDEX IF NOT EXISTS idx_soil_reports_farm_date ON soil_reports (farm_id, report_date DESC);

-- Create crop_recommendations table
CREATE TABLE IF NOT EXISTS crop_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id),
    recommended_crops JSONB NOT NULL,
    confidence_score DECIMAL(5,2),
    reasoning TEXT,
    weather_data JSONB,
    market_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create disease_detections table
CREATE TABLE IF NOT EXISTS disease_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id),
    image_path VARCHAR(500) NOT NULL,
    crop_type VARCHAR(100) NOT NULL,
    detected_disease VARCHAR(255),
    confidence_score DECIMAL(5,2),
    severity_level severity_level,
    treatment_recommendations JSONB,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create irrigation_schedules table
CREATE TABLE IF NOT EXISTS irrigation_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id),
    crop_type VARCHAR(100) NOT NULL,
    scheduled_date TIMESTAMP NOT NULL,
    water_amount_liters DECIMAL(10,2),
    irrigation_method irrigation_method,
    weather_forecast JSONB,
    soil_moisture_level DECIMAL(5,2),
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create market_prices table
CREATE TABLE IF NOT EXISTS market_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    crop_name VARCHAR(100) NOT NULL,
    market_name VARCHAR(255) NOT NULL,
    state VARCHAR(100) NOT NULL,
    min_price DECIMAL(10,2),
    max_price DECIMAL(10,2),
    modal_price DECIMAL(10,2),
    price_date TIMESTAMP NOT NULL,
    unit VARCHAR(50) DEFAULT 'quintal',
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on market_prices
CREATE INDEX IF NOT EXISTS idx_market_prices_crop_date ON market_prices (crop_name, price_date DESC);

-- Create yield_predictions table
CREATE TABLE IF NOT EXISTS yield_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID NOT NULL REFERENCES farms(id),
    crop_type VARCHAR(100) NOT NULL,
    predicted_yield_tons_per_hectare DECIMAL(8,2),
    confidence_interval JSONB,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    harvest_date TIMESTAMP,
    input_features JSONB,
    model_version VARCHAR(50)
);

-- Create weather_data table with spatial support
CREATE TABLE IF NOT EXISTS weather_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    location GEOMETRY(POINT, 4326) NOT NULL,
    temperature_celsius DECIMAL(5,2),
    humidity_percent DECIMAL(5,2),
    rainfall_mm DECIMAL(8,2),
    wind_speed_kmh DECIMAL(6,2),
    pressure_hpa DECIMAL(7,2),
    uv_index DECIMAL(4,2),
    recorded_at TIMESTAMP NOT NULL,
    source VARCHAR(100),
    forecast_data JSONB
);

-- Create spatial index on weather_data
CREATE INDEX IF NOT EXISTS idx_weather_location ON weather_data USING GIST (location);

-- Create voice_queries table
CREATE TABLE IF NOT EXISTS voice_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_id UUID REFERENCES farms(id),
    audio_file_path VARCHAR(500),
    transcribed_text TEXT,
    detected_language VARCHAR(10),
    confidence_score DECIMAL(5,2),
    agent_response TEXT,
    processing_time_seconds DECIMAL(8,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create functions for spatial queries
CREATE OR REPLACE FUNCTION find_nearby_farms(
    center_lat DECIMAL,
    center_lon DECIMAL,
    radius_km DECIMAL DEFAULT 10
)
RETURNS TABLE (
    farm_id UUID,
    farm_name VARCHAR(255),
    distance_km DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.id,
        f.name,
        (ST_Distance(f.location, ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)) * 111.32)::DECIMAL AS distance_km
    FROM farms f
    WHERE ST_DWithin(
        f.location,
        ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326),
        radius_km / 111.32
    )
    ORDER BY distance_km;
END;
$$ LANGUAGE plpgsql;

-- Create function to get weather data for farm
CREATE OR REPLACE FUNCTION get_farm_weather(
    farm_uuid UUID,
    days_back INTEGER DEFAULT 7
)
RETURNS TABLE (
    weather_id UUID,
    temperature DECIMAL,
    humidity DECIMAL,
    rainfall DECIMAL,
    recorded_at TIMESTAMP
) AS $$
DECLARE
    farm_location GEOMETRY;
BEGIN
    -- Get farm location
    SELECT location INTO farm_location FROM farms WHERE id = farm_uuid;
    
    IF farm_location IS NULL THEN
        RAISE EXCEPTION 'Farm not found';
    END IF;
    
    RETURN QUERY
    SELECT 
        w.id,
        w.temperature_celsius,
        w.humidity_percent,
        w.rainfall_mm,
        w.recorded_at
    FROM weather_data w
    WHERE ST_DWithin(w.location, farm_location, 0.1) -- Within ~11km
    AND w.recorded_at >= CURRENT_TIMESTAMP - INTERVAL '%s days' % days_back
    ORDER BY w.recorded_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_farms_updated_at
    BEFORE UPDATE ON farms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data
INSERT INTO farms (name, owner_name, location, area_hectares, soil_type) VALUES
('Green Valley Farm', 'Rajesh Kumar', ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326), 5.5, 'Loamy'),
('Sunrise Agriculture', 'Priya Sharma', ST_SetSRID(ST_MakePoint(77.3910, 28.5355), 4326), 8.2, 'Clay'),
('Golden Fields', 'Amit Patel', ST_SetSRID(ST_MakePoint(72.5714, 23.0225), 4326), 12.0, 'Sandy Loam'),
('Harvest Moon Farm', 'Lakshmi Reddy', ST_SetSRID(ST_MakePoint(78.4867, 17.3850), 4326), 6.8, 'Black Soil'),
('Organic Oasis', 'Suresh Singh', ST_SetSRID(ST_MakePoint(75.7873, 26.9124), 4326), 4.3, 'Alluvial');

-- Insert sample market prices
INSERT INTO market_prices (crop_name, market_name, state, min_price, max_price, modal_price, price_date, source) VALUES
('wheat', 'Delhi Mandi', 'Delhi', 1800, 2200, 2000, CURRENT_DATE - INTERVAL '1 day', 'agmarknet'),
('rice', 'Mumbai Mandi', 'Maharashtra', 2200, 2800, 2500, CURRENT_DATE - INTERVAL '1 day', 'agmarknet'),
('tomato', 'Bangalore Mandi', 'Karnataka', 800, 1200, 1000, CURRENT_DATE - INTERVAL '1 day', 'agmarknet'),
('onion', 'Pune Mandi', 'Maharashtra', 1200, 1600, 1400, CURRENT_DATE - INTERVAL '1 day', 'agmarknet'),
('potato', 'Kolkata Mandi', 'West Bengal', 600, 900, 750, CURRENT_DATE - INTERVAL '1 day', 'agmarknet');

-- Create views for common queries
CREATE OR REPLACE VIEW farm_summary AS
SELECT 
    f.id,
    f.name,
    f.owner_name,
    f.area_hectares,
    f.soil_type,
    ST_X(f.location) as longitude,
    ST_Y(f.location) as latitude,
    COUNT(sr.id) as soil_reports_count,
    MAX(sr.report_date) as latest_soil_report,
    COUNT(dd.id) as disease_detections_count,
    COUNT(cr.id) as crop_recommendations_count
FROM farms f
LEFT JOIN soil_reports sr ON f.id = sr.farm_id
LEFT JOIN disease_detections dd ON f.id = dd.farm_id
LEFT JOIN crop_recommendations cr ON f.id = cr.farm_id
GROUP BY f.id, f.name, f.owner_name, f.area_hectares, f.soil_type, f.location;

-- Create view for recent market trends
CREATE OR REPLACE VIEW market_trends AS
SELECT 
    crop_name,
    market_name,
    state,
    AVG(modal_price) as avg_price,
    MIN(modal_price) as min_price,
    MAX(modal_price) as max_price,
    STDDEV(modal_price) as price_volatility,
    COUNT(*) as data_points,
    MAX(price_date) as latest_date
FROM market_prices
WHERE price_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY crop_name, market_name, state
ORDER BY crop_name, avg_price DESC;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agribotx_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agribotx_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO agribotx_user;