#!/usr/bin/env python3
"""
Database initialization script for AgriBotX Pro
Creates tables and sets up PostGIS extensions
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import Base
from db.spatial_queries import create_spatial_indexes

load_dotenv()

def create_database():
    """Create the database and enable PostGIS extension"""
    database_url = os.getenv('DATABASE_URL', 'postgresql://agribotx_user:password@localhost:5432/agribotx_pro')
    
    # Connect to postgres database to create our database
    postgres_url = database_url.rsplit('/', 1)[0] + '/postgres'
    engine = create_engine(postgres_url)
    
    db_name = database_url.split('/')[-1]
    
    with engine.connect() as conn:
        # Terminate existing connections to the database
        conn.execute(text(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{db_name}' AND pid <> pg_backend_pid()
        """))
        
        # Drop and recreate database
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
        conn.execute(text(f"CREATE DATABASE {db_name}"))
    
    engine.dispose()

def setup_postgis():
    """Enable PostGIS extension and create spatial reference systems"""
    database_url = os.getenv('DATABASE_URL', 'postgresql://agribotx_user:password@localhost:5432/agribotx_pro')
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Enable PostGIS extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology"))
        
        # Create custom spatial reference systems if needed
        conn.execute(text("""
            INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext)
            SELECT 4326, 'EPSG', 4326, 
                   '+proj=longlat +datum=WGS84 +no_defs',
                   'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
            WHERE NOT EXISTS (SELECT 1 FROM spatial_ref_sys WHERE srid = 4326)
        """))
        
        conn.commit()
    
    engine.dispose()

def create_tables():
    """Create all database tables"""
    database_url = os.getenv('DATABASE_URL', 'postgresql://agribotx_user:password@localhost:5432/agribotx_pro')
    engine = create_engine(database_url)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create spatial indexes
    create_spatial_indexes(engine)
    
    engine.dispose()

def insert_sample_data():
    """Insert sample data for testing"""
    database_url = os.getenv('DATABASE_URL', 'postgresql://agribotx_user:password@localhost:5432/agribotx_pro')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Sample farm data
        from db.models import Farm, SoilReport, MarketPrice
        from datetime import datetime, timedelta
        
        # Create sample farm
        sample_farm = Farm(
            name="Green Valley Farm",
            owner_name="Rajesh Kumar",
            location="POINT(77.2090 28.6139)",  # Delhi coordinates
            area_hectares=5.5,
            soil_type="Loamy"
        )
        session.add(sample_farm)
        session.flush()  # Get the farm ID
        
        # Create sample soil report
        soil_report = SoilReport(
            farm_id=sample_farm.id,
            ph_level=6.8,
            nitrogen=0.045,
            phosphorus=22.5,
            potassium=180.0,
            organic_matter=2.8,
            moisture_content=15.2,
            report_date=datetime.utcnow() - timedelta(days=30),
            lab_name="Agricultural Testing Lab Delhi"
        )
        session.add(soil_report)
        
        # Create sample market prices
        crops = ["wheat", "rice", "tomato", "potato", "onion"]
        markets = ["Delhi", "Mumbai", "Kolkata"]
        
        for crop in crops:
            for market in markets:
                price = MarketPrice(
                    crop_name=crop,
                    market_name=f"{market} Mandi",
                    state=market,
                    min_price=1000 + (hash(crop + market) % 500),
                    max_price=1500 + (hash(crop + market) % 800),
                    modal_price=1200 + (hash(crop + market) % 600),
                    price_date=datetime.utcnow() - timedelta(days=1),
                    source="agmarknet"
                )
                session.add(price)
        
        session.commit()
        print("✅ Sample data inserted successfully")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error inserting sample data: {e}")
    finally:
        session.close()
        engine.dispose()

def main():
    """Main initialization function"""
    print("🚀 Initializing AgriBotX Pro Database...")
    
    try:
        print("1. Creating database...")
        create_database()
        
        print("2. Setting up PostGIS...")
        setup_postgis()
        
        print("3. Creating tables...")
        create_tables()
        
        print("4. Inserting sample data...")
        insert_sample_data()
        
        print("✅ Database initialization completed successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()