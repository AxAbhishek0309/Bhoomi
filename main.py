#!/usr/bin/env python3
"""
AgriBotX Pro - Main Entry Point
Multi-agent AI system for precision agriculture
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

def main():
    """Main entry point for AgriBotX Pro"""
    
    parser = argparse.ArgumentParser(
        description="AgriBotX Pro - AI-Powered Agricultural Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode cli                    # Start CLI interface
  python main.py --mode web                    # Start web dashboard
  python main.py --mode init                   # Initialize database
  python main.py --agent crop --farm-id 123   # Quick crop recommendation
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['cli', 'web', 'init', 'agent'],
        default='cli',
        help='Operation mode (default: cli)'
    )
    
    parser.add_argument(
        '--agent',
        choices=['crop', 'soil', 'disease', 'irrigation', 'market', 'yield', 'voice'],
        help='Specific agent to run (for agent mode)'
    )
    
    parser.add_argument(
        '--farm-id',
        help='Farm ID for agent operations'
    )
    
    parser.add_argument(
        '--input-file',
        help='Input file for analysis (soil reports, images, etc.)'
    )
    
    parser.add_argument(
        '--crop-type',
        help='Crop type for analysis'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for web dashboard (default: 7860)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host for web dashboard (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check environment setup
    if not check_environment():
        print("❌ Environment setup incomplete. Please check your .env file.")
        return 1
    
    try:
        if args.mode == 'init':
            return initialize_system()
        
        elif args.mode == 'cli':
            return start_cli()
        
        elif args.mode == 'web':
            return start_web_dashboard(args.host, args.port, args.debug)
        
        elif args.mode == 'agent':
            return run_single_agent(args)
        
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thank you for using AgriBotX Pro.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

def print_banner():
    """Print AgriBotX Pro banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🌾 AgriBotX Pro - AI-Powered Agricultural Assistant 🌾    ║
    ║                                                              ║
    ║     Multi-Agent System for Precision Agriculture            ║
    ║     • Crop Recommendations  • Soil Analysis                 ║
    ║     • Disease Detection     • Irrigation Advice             ║
    ║     • Market Forecasting    • Yield Prediction              ║
    ║     • Voice Interface       • Spatial Analytics             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if environment is properly configured"""
    required_vars = ['OPENAI_API_KEY']
    optional_vars = [
        'OPENWEATHER_API_KEY',
        'SENTINELHUB_CLIENT_ID',
        'AGMARKNET_API_KEY',
        'TWILIO_ACCOUNT_SID',
        'DATABASE_URL'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional environment variables: {', '.join(missing_optional)}")
        print("   Some features may not work properly.")
    
    print("✅ Environment configuration looks good!")
    return True

def initialize_system():
    """Initialize the AgriBotX Pro system"""
    print("🚀 Initializing AgriBotX Pro system...")
    
    try:
        from db.init_db import main as init_db_main
        init_db_main()
        
        # Create sample data directories
        create_sample_directories()
        
        print("✅ System initialization completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

def create_sample_directories():
    """Create sample data directories and files"""
    directories = [
        'data/mock_soil_reports',
        'data/crop_images',
        'data/ndvi_samples',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create sample soil report
    sample_soil_report = {
        "farm_id": "sample_farm_001",
        "ph_level": 6.8,
        "nitrogen": 0.045,
        "phosphorus": 22.5,
        "potassium": 180.0,
        "organic_matter": 2.8,
        "moisture_content": 15.2,
        "lab_name": "Agricultural Testing Lab",
        "report_date": "2024-01-15",
        "location": {
            "latitude": 28.6139,
            "longitude": 77.2090
        }
    }
    
    import json
    with open('data/mock_soil_reports/sample_report.json', 'w') as f:
        json.dump(sample_soil_report, f, indent=2)
    
    print("📁 Sample data directories created")

def start_cli():
    """Start the CLI interface"""
    print("🖥️  Starting CLI interface...")
    
    try:
        from cli.agribotx_cli import cli
        cli()
        return 0
        
    except Exception as e:
        print(f"❌ CLI startup failed: {e}")
        return 1

def start_web_dashboard(host, port, debug):
    """Start the web dashboard"""
    print(f"🌐 Starting web dashboard on http://{host}:{port}")
    
    try:
        from ui.dashboard import create_dashboard
        
        demo = create_dashboard()
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,
            debug=debug
        )
        return 0
        
    except Exception as e:
        print(f"❌ Web dashboard startup failed: {e}")
        return 1

def run_single_agent(args):
    """Run a single agent for quick operations"""
    print(f"🤖 Running {args.agent} agent...")
    
    if not args.farm_id and args.agent in ['crop', 'irrigation', 'yield']:
        print("❌ Farm ID required for this agent")
        return 1
    
    try:
        if args.agent == 'crop':
            return run_crop_agent(args.farm_id)
        
        elif args.agent == 'soil':
            if not args.input_file:
                print("❌ Input file required for soil analysis")
                return 1
            return run_soil_agent(args.input_file, args.farm_id)
        
        elif args.agent == 'disease':
            if not args.input_file or not args.crop_type:
                print("❌ Input file and crop type required for disease detection")
                return 1
            return run_disease_agent(args.input_file, args.crop_type)
        
        elif args.agent == 'market':
            if not args.crop_type:
                print("❌ Crop type required for market analysis")
                return 1
            return run_market_agent(args.crop_type)
        
        else:
            print(f"❌ Agent '{args.agent}' not implemented for single mode")
            return 1
            
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return 1

def run_crop_agent(farm_id):
    """Run crop recommendation agent"""
    try:
        from agents.crop_recommender import CropRecommenderAgent
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Initialize agent
        agent = CropRecommenderAgent({})
        
        # Get database session
        database_url = os.getenv('DATABASE_URL', 'sqlite:///agribotx.db')
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get recommendations
        result = asyncio.run(agent.recommend_crops(farm_id, session))
        
        # Display results
        print("\n🌱 Crop Recommendations:")
        print("=" * 50)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return 1
        
        if 'recommendations' in result:
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n{i}. {rec.get('crop_name', 'Unknown').title()}")
                print(f"   Suitability: {rec.get('suitability_score', 0)}/100")
                print(f"   Expected Yield: {rec.get('expected_yield', 'N/A')}")
                print(f"   Market Potential: {rec.get('market_potential', 'N/A')}")
        
        if 'general_advice' in result:
            print(f"\n💡 General Advice:\n{result['general_advice']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Crop agent error: {e}")
        return 1

def run_soil_agent(input_file, farm_id):
    """Run soil analysis agent"""
    try:
        from agents.soil_analyzer import SoilAnalyzerAgent
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return 1
        
        # Initialize agent
        agent = SoilAnalyzerAgent({})
        
        # Get database session
        database_url = os.getenv('DATABASE_URL', 'sqlite:///agribotx.db')
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Analyze soil
        result = asyncio.run(agent.analyze_soil_report(input_file, farm_id, session))
        
        # Display results
        print("\n🌍 Soil Analysis Results:")
        print("=" * 50)
        print(f"Overall Health Score: {result.overall_health_score:.1f}/100")
        
        # pH Analysis
        ph_info = result.ph_analysis
        print(f"\npH Level: {ph_info.get('value', 'N/A')} ({ph_info.get('status', 'Unknown')})")
        
        # Nutrient Analysis
        if result.nutrient_analysis:
            print("\nNutrient Analysis:")
            for nutrient in result.nutrient_analysis:
                print(f"  {nutrient.nutrient.title()}: {nutrient.value} {nutrient.unit} ({nutrient.status})")
        
        # Recommendations
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Soil agent error: {e}")
        return 1

def run_disease_agent(input_file, crop_type):
    """Run disease detection agent"""
    try:
        from agents.disease_detector import DiseaseDetectorAgent
        
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return 1
        
        # Initialize agent
        agent = DiseaseDetectorAgent({})
        
        # Detect disease
        result = asyncio.run(agent.detect_disease(input_file, crop_type))
        
        # Display results
        print("\n🦠 Disease Detection Results:")
        print("=" * 50)
        print(f"Detected Disease: {result.disease_name.replace('_', ' ').title()}")
        print(f"Confidence: {result.confidence_score:.1%}")
        print(f"Severity: {result.severity_level.title()}")
        
        if result.treatment_recommendations:
            print("\nTreatment Recommendations:")
            for rec in result.treatment_recommendations:
                print(f"  • {rec}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Disease agent error: {e}")
        return 1

def run_market_agent(crop_type):
    """Run market analysis agent"""
    try:
        from agents.market_forecaster import MarketForecastAgent
        
        # Initialize agent
        agent = MarketForecastAgent({})
        
        # Get current prices
        prices = asyncio.run(agent.get_current_prices(crop_type))
        
        # Display results
        print(f"\n📈 Current Prices for {crop_type.title()}:")
        print("=" * 50)
        
        if not prices:
            print("No price data available")
            return 1
        
        for price in prices:
            print(f"\n{price.get('market_name', 'Unknown Market')}:")
            print(f"  Min Price: ₹{price.get('min_price', 0):.0f}/quintal")
            print(f"  Max Price: ₹{price.get('max_price', 0):.0f}/quintal")
            print(f"  Modal Price: ₹{price.get('modal_price', 0):.0f}/quintal")
        
        return 0
        
    except Exception as e:
        print(f"❌ Market agent error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())