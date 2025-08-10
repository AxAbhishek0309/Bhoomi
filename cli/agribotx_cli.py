#!/usr/bin/env python3
"""
AgriBotX Pro Command Line Interface
Provides CLI access to all agricultural AI agents
"""

import os
import sys
import click
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.crop_recommender import CropRecommenderAgent
from agents.soil_analyzer import SoilAnalyzerAgent
from agents.disease_detector import DiseaseDetectorAgent
from agents.irrigation_advisor import IrrigationAdvisorAgent
from agents.market_forecaster import MarketForecastAgent
from agents.yield_predictor import YieldPredictorAgent
from agents.voice_interface import VoiceInterfaceAgent

from db.models import Base, Farm
from db.init_db import create_database, setup_postgis, create_tables
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

console = Console()

class AgriBotXCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.config = self._load_config()
        self.session = None
        self.agents = {}
        self._initialize_agents()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = Path(__file__).parent.parent / 'configs' / 'agent_configs.yaml'
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return {}
    
    def _initialize_agents(self):
        """Initialize all AI agents"""
        try:
            agent_configs = self.config.get('agents', {})
            
            self.agents = {
                'crop_recommender': CropRecommenderAgent(agent_configs.get('crop_recommender', {})),
                'soil_analyzer': SoilAnalyzerAgent(agent_configs.get('soil_analyzer', {})),
                'disease_detector': DiseaseDetectorAgent(agent_configs.get('disease_detector', {})),
                'irrigation_advisor': IrrigationAdvisorAgent(agent_configs.get('irrigation_advisor', {})),
                'market_forecaster': MarketForecastAgent(agent_configs.get('market_forecaster', {})),
                'yield_predictor': YieldPredictorAgent(agent_configs.get('yield_predictor', {})),
                'voice_interface': VoiceInterfaceAgent(agent_configs.get('voice_interface', {}))
            }
            
        except Exception as e:
            console.print(f"[red]Error initializing agents: {e}[/red]")
            self.agents = {}
    
    def _get_database_session(self):
        """Get database session"""
        if self.session is None:
            try:
                database_url = os.getenv('DATABASE_URL', 'sqlite:///agribotx.db')
                engine = create_engine(database_url)
                Session = sessionmaker(bind=engine)
                self.session = Session()
            except Exception as e:
                console.print(f"[red]Database connection error: {e}[/red]")
                return None
        return self.session

@click.group()
@click.version_option(version='1.0.0', prog_name='AgriBotX Pro')
def cli():
    """🌾 AgriBotX Pro - AI-Powered Agricultural Assistant"""
    pass

@cli.command()
def init():
    """Initialize the AgriBotX Pro database and setup"""
    console.print(Panel.fit("🚀 Initializing AgriBotX Pro", style="bold green"))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task1 = progress.add_task("Creating database...", total=None)
            create_database()
            progress.update(task1, completed=True)
            
            task2 = progress.add_task("Setting up PostGIS...", total=None)
            setup_postgis()
            progress.update(task2, completed=True)
            
            task3 = progress.add_task("Creating tables...", total=None)
            create_tables()
            progress.update(task3, completed=True)
        
        console.print("✅ [green]AgriBotX Pro initialized successfully![/green]")
        
    except Exception as e:
        console.print(f"❌ [red]Initialization failed: {e}[/red]")

@cli.group()
def crop():
    """🌱 Crop recommendation and analysis"""
    pass

@crop.command('recommend')
@click.option('--farm-id', required=True, help='Farm ID for recommendations')
@click.option('--crop-type', help='Specific crop type to analyze')
@click.option('--season', help='Growing season (kharif/rabi/zaid)')
def crop_recommend(farm_id, crop_type, season):
    """Get crop recommendations for a farm"""
    console.print(Panel.fit("🌱 Generating Crop Recommendations", style="bold blue"))
    
    try:
        app = AgriBotXCLI()
        session = app._get_database_session()
        
        if not session:
            return
        
        agent = app.agents.get('crop_recommender')
        if not agent:
            console.print("[red]Crop recommender agent not available[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing farm conditions...", total=None)
            
            # Run async function
            result = asyncio.run(agent.recommend_crops(farm_id, session))
            progress.update(task, completed=True)
        
        if 'error' in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return
        
        # Display recommendations
        if 'recommendations' in result:
            table = Table(title="Crop Recommendations")
            table.add_column("Crop", style="cyan")
            table.add_column("Suitability", style="green")
            table.add_column("Expected Yield", style="yellow")
            table.add_column("Market Potential", style="magenta")
            
            for rec in result['recommendations']:
                table.add_row(
                    rec.get('crop_name', 'N/A'),
                    f"{rec.get('suitability_score', 0)}/100",
                    rec.get('expected_yield', 'N/A'),
                    rec.get('market_potential', 'N/A')
                )
            
            console.print(table)
        
        # Display general advice
        if 'general_advice' in result:
            console.print(Panel(result['general_advice'], title="General Advice", style="blue"))
        
    except Exception as e:
        console.print(f"[red]Error generating recommendations: {e}[/red]")

@cli.group()
def soil():
    """🌍 Soil analysis and recommendations"""
    pass

@soil.command('analyze')
@click.option('--file', '-f', required=True, help='Path to soil report file')
@click.option('--farm-id', required=True, help='Farm ID')
@click.option('--format', default='auto', help='File format (json/csv/xml/auto)')
def soil_analyze(file, farm_id, format):
    """Analyze soil report and provide recommendations"""
    console.print(Panel.fit("🌍 Analyzing Soil Report", style="bold brown"))
    
    try:
        if not os.path.exists(file):
            console.print(f"[red]File not found: {file}[/red]")
            return
        
        app = AgriBotXCLI()
        session = app._get_database_session()
        
        if not session:
            return
        
        agent = app.agents.get('soil_analyzer')
        if not agent:
            console.print("[red]Soil analyzer agent not available[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing soil data...", total=None)
            
            result = asyncio.run(agent.analyze_soil_report(file, farm_id, session))
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"[green]Soil Health Score: {result.overall_health_score:.1f}/100[/green]")
        
        # pH Analysis
        ph_info = result.ph_analysis
        console.print(f"pH Level: {ph_info.get('value', 'N/A')} ({ph_info.get('status', 'Unknown')})")
        
        # Nutrient Analysis
        if result.nutrient_analysis:
            table = Table(title="Nutrient Analysis")
            table.add_column("Nutrient", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Recommendation", style="blue")
            
            for nutrient in result.nutrient_analysis:
                table.add_row(
                    nutrient.nutrient.title(),
                    f"{nutrient.value} {nutrient.unit}",
                    nutrient.status.title(),
                    nutrient.recommendation
                )
            
            console.print(table)
        
        # Recommendations
        if result.recommendations:
            console.print(Panel("\n".join(f"• {rec}" for rec in result.recommendations), 
                               title="Recommendations", style="green"))
        
        # Deficiencies
        if result.deficiencies:
            console.print(Panel("\n".join(f"• {def_}" for def_ in result.deficiencies), 
                               title="Deficiencies Found", style="red"))
        
    except Exception as e:
        console.print(f"[red]Error analyzing soil: {e}[/red]")

@cli.group()
def disease():
    """🦠 Plant disease detection"""
    pass

@disease.command('detect')
@click.option('--image', '-i', required=True, help='Path to plant image')
@click.option('--crop-type', required=True, help='Type of crop in image')
@click.option('--farm-id', help='Farm ID (optional)')
def disease_detect(image, crop_type, farm_id):
    """Detect diseases in plant images"""
    console.print(Panel.fit("🦠 Detecting Plant Diseases", style="bold red"))
    
    try:
        if not os.path.exists(image):
            console.print(f"[red]Image file not found: {image}[/red]")
            return
        
        app = AgriBotXCLI()
        session = app._get_database_session() if farm_id else None
        
        agent = app.agents.get('disease_detector')
        if not agent:
            console.print("[red]Disease detector agent not available[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing plant image...", total=None)
            
            result = asyncio.run(agent.detect_disease(image, crop_type, farm_id, session))
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"[yellow]Detected Disease: {result.disease_name.replace('_', ' ').title()}[/yellow]")
        console.print(f"[green]Confidence: {result.confidence_score:.2%}[/green]")
        console.print(f"[red]Severity: {result.severity_level.title()}[/red]")
        
        if result.affected_area_percentage:
            console.print(f"[orange1]Affected Area: {result.affected_area_percentage:.1f}%[/orange1]")
        
        # Treatment recommendations
        if result.treatment_recommendations:
            console.print(Panel("\n".join(f"• {rec}" for rec in result.treatment_recommendations), 
                               title="Treatment Recommendations", style="blue"))
        
        # Prevention measures
        if result.prevention_measures:
            console.print(Panel("\n".join(f"• {prev}" for prev in result.prevention_measures), 
                               title="Prevention Measures", style="green"))
        
        # Economic impact
        if result.economic_impact:
            console.print(Panel(result.economic_impact, title="Economic Impact", style="yellow"))
        
    except Exception as e:
        console.print(f"[red]Error detecting disease: {e}[/red]")

@cli.group()
def irrigation():
    """💧 Irrigation scheduling and advice"""
    pass

@irrigation.command('schedule')
@click.option('--farm-id', required=True, help='Farm ID')
@click.option('--crop-type', required=True, help='Type of crop')
@click.option('--growth-stage', required=True, help='Current growth stage')
def irrigation_schedule(farm_id, crop_type, growth_stage):
    """Get irrigation schedule recommendations"""
    console.print(Panel.fit("💧 Generating Irrigation Schedule", style="bold blue"))
    
    try:
        app = AgriBotXCLI()
        session = app._get_database_session()
        
        if not session:
            return
        
        agent = app.agents.get('irrigation_advisor')
        if not agent:
            console.print("[red]Irrigation advisor agent not available[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing irrigation needs...", total=None)
            
            result = asyncio.run(agent.get_irrigation_recommendation(farm_id, crop_type, growth_stage, session))
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"[blue]Scheduled Date: {result.scheduled_date.strftime('%Y-%m-%d %H:%M')}[/blue]")
        console.print(f"[cyan]Water Amount: {result.water_amount_liters:.0f} liters[/cyan]")
        console.print(f"[green]Method: {result.irrigation_method.title()}[/green]")
        console.print(f"[yellow]Duration: {result.duration_minutes} minutes[/yellow]")
        console.print(f"[red]Urgency: {result.urgency_level.title()}[/red]")
        
        # Reasoning
        console.print(Panel(result.reasoning, title="Reasoning", style="blue"))
        
        # Weather considerations
        if result.weather_considerations:
            console.print(Panel("\n".join(f"• {cons}" for cons in result.weather_considerations), 
                               title="Weather Considerations", style="yellow"))
        
    except Exception as e:
        console.print(f"[red]Error generating irrigation schedule: {e}[/red]")

@cli.group()
def market():
    """📈 Market prices and forecasting"""
    pass

@market.command('prices')
@click.option('--crop', required=True, help='Crop name')
@click.option('--market', help='Market name (optional)')
def market_prices(crop, market):
    """Get current market prices for crops"""
    console.print(Panel.fit("📈 Fetching Market Prices", style="bold green"))
    
    try:
        app = AgriBotXCLI()
        agent = app.agents.get('market_forecaster')
        
        if not agent:
            console.print("[red]Market forecaster agent not available[/red]")
            return
        
        markets = [market] if market else None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching price data...", total=None)
            
            prices = asyncio.run(agent.get_current_prices(crop, markets))
            progress.update(task, completed=True)
        
        if not prices:
            console.print("[yellow]No price data available[/yellow]")
            return
        
        # Display prices
        table = Table(title=f"Current Prices for {crop.title()}")
        table.add_column("Market", style="cyan")
        table.add_column("Min Price", style="green")
        table.add_column("Max Price", style="red")
        table.add_column("Modal Price", style="yellow")
        table.add_column("Date", style="blue")
        
        for price in prices:
            table.add_row(
                price.get('market_name', 'N/A'),
                f"₹{price.get('min_price', 0):.0f}",
                f"₹{price.get('max_price', 0):.0f}",
                f"₹{price.get('modal_price', 0):.0f}",
                price.get('price_date', datetime.now()).strftime('%Y-%m-%d') if isinstance(price.get('price_date'), datetime) else str(price.get('price_date', 'N/A'))
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error fetching market prices: {e}[/red]")

@market.command('forecast')
@click.option('--crop', required=True, help='Crop name')
@click.option('--market', required=True, help='Market name')
@click.option('--days', default=30, help='Forecast period in days')
def market_forecast(crop, market, days):
    """Get price forecast for a crop in a specific market"""
    console.print(Panel.fit("🔮 Generating Price Forecast", style="bold magenta"))
    
    try:
        app = AgriBotXCLI()
        agent = app.agents.get('market_forecaster')
        
        if not agent:
            console.print("[red]Market forecaster agent not available[/red]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating forecast...", total=None)
            
            forecast = asyncio.run(agent.forecast_prices(crop, market, days))
            progress.update(task, completed=True)
        
        # Display forecast
        console.print(f"[green]Current Price: ₹{forecast.current_price:.0f}[/green]")
        console.print(f"[yellow]Predicted Price: ₹{forecast.predicted_price:.0f}[/yellow]")
        console.print(f"[blue]Price Trend: {forecast.price_trend.title()}[/blue]")
        console.print(f"[cyan]Confidence: {forecast.confidence_level:.1%}[/cyan]")
        console.print(f"[magenta]Forecast Period: {forecast.forecast_period_days} days[/magenta]")
        
        # Factors affecting price
        if forecast.factors_affecting_price:
            console.print(Panel("\n".join(f"• {factor}" for factor in forecast.factors_affecting_price), 
                               title="Factors Affecting Price", style="yellow"))
        
    except Exception as e:
        console.print(f"[red]Error generating forecast: {e}[/red]")

@cli.group()
def voice():
    """🎤 Voice interface for queries"""
    pass

@voice.command('query')
@click.option('--text', help='Text query (for testing without audio)')
@click.option('--audio', help='Path to audio file')
@click.option('--farm-id', help='Farm ID for context')
def voice_query(text, audio, farm_id):
    """Process voice or text queries"""
    console.print(Panel.fit("🎤 Processing Voice Query", style="bold purple"))
    
    try:
        app = AgriBotXCLI()
        agent = app.agents.get('voice_interface')
        
        if not agent:
            console.print("[red]Voice interface agent not available[/red]")
            return
        
        if text:
            # Process text query
            response = agent.process_text_query(text, farm_id)
            console.print(Panel(response, title="Response", style="green"))
        
        elif audio:
            if not os.path.exists(audio):
                console.print(f"[red]Audio file not found: {audio}[/red]")
                return
            
            session = app._get_database_session()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing audio...", total=None)
                
                result = asyncio.run(agent.process_voice_query(audio, farm_id, session))
                progress.update(task, completed=True)
            
            # Display results
            console.print(f"[cyan]Transcribed Text: {result.transcribed_text}[/cyan]")
            console.print(f"[yellow]Detected Language: {result.detected_language}[/yellow]")
            console.print(f"[green]Confidence: {result.confidence_score:.1%}[/green]")
            console.print(f"[blue]Intent: {result.intent}[/blue]")
            console.print(f"[magenta]Processing Time: {result.processing_time_seconds:.2f}s[/magenta]")
            
            console.print(Panel(result.agent_response, title="Response", style="green"))
        
        else:
            console.print("[red]Please provide either --text or --audio option[/red]")
        
    except Exception as e:
        console.print(f"[red]Error processing voice query: {e}[/red]")

@cli.command()
def status():
    """Show system status and agent availability"""
    console.print(Panel.fit("📊 AgriBotX Pro System Status", style="bold cyan"))
    
    try:
        app = AgriBotXCLI()
        
        # Check database connection
        session = app._get_database_session()
        db_status = "✅ Connected" if session else "❌ Not Connected"
        
        # Check agent status
        agent_status = {}
        for name, agent in app.agents.items():
            agent_status[name] = "✅ Available" if agent else "❌ Not Available"
        
        # Display status table
        table = Table(title="System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row("Database", db_status)
        
        for agent_name, status in agent_status.items():
            table.add_row(agent_name.replace('_', ' ').title(), status)
        
        console.print(table)
        
        # Environment check
        env_vars = [
            'OPENAI_API_KEY',
            'OPENWEATHER_API_KEY',
            'SENTINELHUB_CLIENT_ID',
            'AGMARKNET_API_KEY',
            'TWILIO_ACCOUNT_SID'
        ]
        
        env_table = Table(title="Environment Variables")
        env_table.add_column("Variable", style="cyan")
        env_table.add_column("Status", style="green")
        
        for var in env_vars:
            status = "✅ Set" if os.getenv(var) else "❌ Not Set"
            env_table.add_row(var, status)
        
        console.print(env_table)
        
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")

@cli.command()
def api_status():
    """Check live API connectivity and status"""
    console.print(Panel.fit("🔍 Live API Status Check", style="bold blue"))
    
    try:
        from utils.api_status import APIStatusChecker
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Checking all APIs...", total=None)
            
            checker = APIStatusChecker()
            statuses = asyncio.run(checker.check_all_apis())
            
            progress.update(task, completed=True)
        
        # Display detailed API status
        api_table = Table(title="Live API Status")
        api_table.add_column("API", style="cyan")
        api_table.add_column("Status", style="green")
        api_table.add_column("Response Time", style="yellow")
        api_table.add_column("Notes", style="blue")
        
        for api_name, status in statuses.items():
            status_icon = {
                'online': '✅ Online',
                'offline': '❌ Offline',
                'error': '⚠️ Error',
                'no_key': '🔑 No Key'
            }.get(status.status, '❓ Unknown')
            
            response_time = f"{status.response_time_ms:.0f}ms" if status.response_time_ms > 0 else "N/A"
            notes = status.error_message[:30] + "..." if status.error_message else "OK"
            
            api_table.add_row(status.name, status_icon, response_time, notes)
        
        console.print(api_table)
        
        # Summary
        online_count = sum(1 for s in statuses.values() if s.status == 'online')
        total_count = len(statuses)
        
        if online_count == total_count:
            console.print(Panel("🎉 All APIs are operational!", style="green"))
        else:
            console.print(Panel(f"⚠️ {online_count}/{total_count} APIs online. Check configuration for offline APIs.", style="yellow"))
        
    except Exception as e:
        console.print(f"[red]Error checking API status: {e}[/red]")

if __name__ == '__main__':
    cli()