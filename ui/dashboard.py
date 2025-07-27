#!/usr/bin/env python3
"""
AgriBotX Pro Web Dashboard
Gradio-based web interface for agricultural AI agents
"""

import os
import sys
import gradio as gr
import asyncio
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any

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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class AgriBotXDashboard:
    """Main dashboard application class"""
    
    def __init__(self):
        self.config = self._load_config()
        self.session = None
        self.agents = {}
        self._initialize_agents()
        self._initialize_database()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            config_path = Path(__file__).parent.parent / 'configs' / 'agent_configs.yaml'
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
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
            print(f"Error initializing agents: {e}")
            self.agents = {}
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///agribotx.db')
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            self.session = Session()
        except Exception as e:
            print(f"Database connection error: {e}")
            self.session = None
    
    def get_crop_recommendations(self, farm_id: str, crop_type: str = None) -> str:
        """Get crop recommendations for a farm"""
        try:
            if not farm_id:
                return "Please provide a farm ID"
            
            if not self.session:
                return "Database connection not available"
            
            agent = self.agents.get('crop_recommender')
            if not agent:
                return "Crop recommender agent not available"
            
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.recommend_crops(farm_id, self.session))
            loop.close()
            
            if 'error' in result:
                return f"Error: {result['error']}"
            
            # Format recommendations
            output = "🌱 **Crop Recommendations**\n\n"
            
            if 'recommendations' in result:
                for i, rec in enumerate(result['recommendations'], 1):
                    output += f"**{i}. {rec.get('crop_name', 'Unknown').title()}**\n"
                    output += f"   - Suitability Score: {rec.get('suitability_score', 0)}/100\n"
                    output += f"   - Expected Yield: {rec.get('expected_yield', 'N/A')}\n"
                    output += f"   - Market Potential: {rec.get('market_potential', 'N/A')}\n"
                    output += f"   - Reasoning: {rec.get('reasoning', 'N/A')}\n\n"
            
            if 'general_advice' in result:
                output += f"**General Advice:**\n{result['general_advice']}\n\n"
            
            if 'soil_improvements' in result:
                output += f"**Soil Improvements:**\n{result['soil_improvements']}"
            
            return output
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def analyze_soil_report(self, file, farm_id: str) -> str:
        """Analyze uploaded soil report"""
        try:
            if not file:
                return "Please upload a soil report file"
            
            if not farm_id:
                return "Please provide a farm ID"
            
            if not self.session:
                return "Database connection not available"
            
            agent = self.agents.get('soil_analyzer')
            if not agent:
                return "Soil analyzer agent not available"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.analyze_soil_report(file.name, farm_id, self.session))
            loop.close()
            
            # Format results
            output = "🌍 **Soil Analysis Results**\n\n"
            output += f"**Overall Health Score: {result.overall_health_score:.1f}/100**\n\n"
            
            # pH Analysis
            ph_info = result.ph_analysis
            output += f"**pH Analysis:**\n"
            output += f"- Level: {ph_info.get('value', 'N/A')} ({ph_info.get('status', 'Unknown')})\n"
            output += f"- Recommendation: {ph_info.get('recommendation', 'N/A')}\n\n"
            
            # Nutrient Analysis
            if result.nutrient_analysis:
                output += "**Nutrient Analysis:**\n"
                for nutrient in result.nutrient_analysis:
                    output += f"- {nutrient.nutrient.title()}: {nutrient.value} {nutrient.unit} ({nutrient.status})\n"
                    output += f"  Recommendation: {nutrient.recommendation}\n"
                output += "\n"
            
            # Recommendations
            if result.recommendations:
                output += "**Recommendations:**\n"
                for rec in result.recommendations:
                    output += f"• {rec}\n"
                output += "\n"
            
            # Deficiencies
            if result.deficiencies:
                output += "**Deficiencies Found:**\n"
                for def_ in result.deficiencies:
                    output += f"• {def_}\n"
                output += "\n"
            
            # Amendments
            if result.amendments_needed:
                output += "**Recommended Amendments:**\n"
                for amendment in result.amendments_needed:
                    output += f"• {amendment}\n"
            
            return output
            
        except Exception as e:
            return f"Error analyzing soil report: {str(e)}"
    
    def detect_plant_disease(self, image, crop_type: str) -> str:
        """Detect diseases in plant images"""
        try:
            if not image:
                return "Please upload a plant image"
            
            if not crop_type:
                return "Please specify the crop type"
            
            agent = self.agents.get('disease_detector')
            if not agent:
                return "Disease detector agent not available"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.detect_disease(image, crop_type))
            loop.close()
            
            # Format results
            output = "🦠 **Disease Detection Results**\n\n"
            output += f"**Detected Disease:** {result.disease_name.replace('_', ' ').title()}\n"
            output += f"**Confidence:** {result.confidence_score:.1%}\n"
            output += f"**Severity:** {result.severity_level.title()}\n"
            
            if result.affected_area_percentage:
                output += f"**Affected Area:** {result.affected_area_percentage:.1f}%\n"
            
            output += "\n"
            
            # Treatment recommendations
            if result.treatment_recommendations:
                output += "**Treatment Recommendations:**\n"
                for rec in result.treatment_recommendations:
                    output += f"• {rec}\n"
                output += "\n"
            
            # Prevention measures
            if result.prevention_measures:
                output += "**Prevention Measures:**\n"
                for prev in result.prevention_measures:
                    output += f"• {prev}\n"
                output += "\n"
            
            # Economic impact
            if result.economic_impact:
                output += f"**Economic Impact:** {result.economic_impact}\n"
            
            return output
            
        except Exception as e:
            return f"Error detecting disease: {str(e)}"
    
    def get_irrigation_schedule(self, farm_id: str, crop_type: str, growth_stage: str) -> str:
        """Get irrigation schedule recommendations"""
        try:
            if not all([farm_id, crop_type, growth_stage]):
                return "Please provide farm ID, crop type, and growth stage"
            
            if not self.session:
                return "Database connection not available"
            
            agent = self.agents.get('irrigation_advisor')
            if not agent:
                return "Irrigation advisor agent not available"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                agent.get_irrigation_recommendation(farm_id, crop_type, growth_stage, self.session)
            )
            loop.close()
            
            # Format results
            output = "💧 **Irrigation Schedule**\n\n"
            output += f"**Scheduled Date:** {result.scheduled_date.strftime('%Y-%m-%d %H:%M')}\n"
            output += f"**Water Amount:** {result.water_amount_liters:.0f} liters\n"
            output += f"**Method:** {result.irrigation_method.title()}\n"
            output += f"**Duration:** {result.duration_minutes} minutes\n"
            output += f"**Urgency:** {result.urgency_level.title()}\n\n"
            
            output += f"**Reasoning:** {result.reasoning}\n\n"
            
            if result.weather_considerations:
                output += "**Weather Considerations:**\n"
                for cons in result.weather_considerations:
                    output += f"• {cons}\n"
            
            return output
            
        except Exception as e:
            return f"Error generating irrigation schedule: {str(e)}"
    
    def get_market_prices(self, crop_name: str, market_name: str = None) -> tuple:
        """Get current market prices and create visualization"""
        try:
            if not crop_name:
                return "Please provide a crop name", None
            
            agent = self.agents.get('market_forecaster')
            if not agent:
                return "Market forecaster agent not available", None
            
            markets = [market_name] if market_name else None
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            prices = loop.run_until_complete(agent.get_current_prices(crop_name, markets))
            loop.close()
            
            if not prices:
                return "No price data available", None
            
            # Format text output
            output = f"📈 **Current Prices for {crop_name.title()}**\n\n"
            
            for price in prices:
                output += f"**{price.get('market_name', 'Unknown Market')}**\n"
                output += f"- Min Price: ₹{price.get('min_price', 0):.0f}/quintal\n"
                output += f"- Max Price: ₹{price.get('max_price', 0):.0f}/quintal\n"
                output += f"- Modal Price: ₹{price.get('modal_price', 0):.0f}/quintal\n"
                
                price_date = price.get('price_date')
                if isinstance(price_date, datetime):
                    output += f"- Date: {price_date.strftime('%Y-%m-%d')}\n"
                else:
                    output += f"- Date: {price_date}\n"
                output += "\n"
            
            # Create visualization
            df = pd.DataFrame(prices)
            
            fig = go.Figure()
            
            markets = df['market_name'].unique()
            for market in markets:
                market_data = df[df['market_name'] == market]
                
                fig.add_trace(go.Bar(
                    name=f"{market} - Min",
                    x=[market],
                    y=market_data['min_price'],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name=f"{market} - Modal",
                    x=[market],
                    y=market_data['modal_price'],
                    marker_color='blue'
                ))
                
                fig.add_trace(go.Bar(
                    name=f"{market} - Max",
                    x=[market],
                    y=market_data['max_price'],
                    marker_color='darkblue'
                ))
            
            fig.update_layout(
                title=f"Current Prices for {crop_name.title()}",
                xaxis_title="Market",
                yaxis_title="Price (₹/quintal)",
                barmode='group'
            )
            
            return output, fig
            
        except Exception as e:
            return f"Error fetching market prices: {str(e)}", None
    
    def get_price_forecast(self, crop_name: str, market_name: str, days: int = 30) -> str:
        """Get price forecast for a crop"""
        try:
            if not all([crop_name, market_name]):
                return "Please provide crop name and market name"
            
            agent = self.agents.get('market_forecaster')
            if not agent:
                return "Market forecaster agent not available"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            forecast = loop.run_until_complete(
                agent.forecast_prices(crop_name, market_name, days)
            )
            loop.close()
            
            # Format results
            output = f"🔮 **Price Forecast for {crop_name.title()} in {market_name.title()}**\n\n"
            output += f"**Current Price:** ₹{forecast.current_price:.0f}/quintal\n"
            output += f"**Predicted Price:** ₹{forecast.predicted_price:.0f}/quintal\n"
            
            price_change = forecast.predicted_price - forecast.current_price
            change_percent = (price_change / forecast.current_price) * 100
            
            if price_change > 0:
                output += f"**Expected Change:** +₹{price_change:.0f} (+{change_percent:.1f}%) ⬆️\n"
            elif price_change < 0:
                output += f"**Expected Change:** ₹{price_change:.0f} ({change_percent:.1f}%) ⬇️\n"
            else:
                output += f"**Expected Change:** No significant change ➡️\n"
            
            output += f"**Price Trend:** {forecast.price_trend.title()}\n"
            output += f"**Confidence:** {forecast.confidence_level:.1%}\n"
            output += f"**Forecast Period:** {forecast.forecast_period_days} days\n\n"
            
            if forecast.factors_affecting_price:
                output += "**Factors Affecting Price:**\n"
                for factor in forecast.factors_affecting_price:
                    output += f"• {factor}\n"
            
            return output
            
        except Exception as e:
            return f"Error generating price forecast: {str(e)}"
    
    def predict_yield(self, farm_id: str, crop_type: str, planting_date: str) -> str:
        """Predict crop yield for a farm"""
        try:
            if not all([farm_id, crop_type, planting_date]):
                return "Please provide farm ID, crop type, and planting date"
            
            if not self.session:
                return "Database connection not available"
            
            agent = self.agents.get('yield_predictor')
            if not agent:
                return "Yield predictor agent not available"
            
            # Parse planting date
            try:
                planting_dt = datetime.strptime(planting_date, '%Y-%m-%d')
            except ValueError:
                return "Invalid date format. Please use YYYY-MM-DD"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                agent.predict_yield(farm_id, crop_type, planting_dt, self.session)
            )
            loop.close()
            
            # Format results
            output = f"📊 **Yield Prediction for {crop_type.title()}**\n\n"
            output += f"**Predicted Yield:** {result.predicted_yield_tons_per_hectare:.2f} tons/hectare\n"
            output += f"**Confidence Interval:** {result.confidence_interval[0]:.2f} - {result.confidence_interval[1]:.2f} tons/hectare\n"
            output += f"**Prediction Accuracy:** {result.prediction_accuracy:.1%}\n"
            output += f"**Estimated Harvest Date:** {result.harvest_date_estimate.strftime('%Y-%m-%d')}\n\n"
            
            if result.key_factors:
                output += "**Key Factors:**\n"
                for factor in result.key_factors:
                    output += f"• {factor}\n"
                output += "\n"
            
            if result.risk_factors:
                output += "**Risk Factors:**\n"
                for risk in result.risk_factors:
                    output += f"• {risk}\n"
                output += "\n"
            
            if result.recommendations:
                output += "**Recommendations:**\n"
                for rec in result.recommendations:
                    output += f"• {rec}\n"
            
            return output
            
        except Exception as e:
            return f"Error predicting yield: {str(e)}"
    
    def process_voice_query(self, audio, text_input: str, farm_id: str = None) -> str:
        """Process voice or text queries"""
        try:
            agent = self.agents.get('voice_interface')
            if not agent:
                return "Voice interface agent not available"
            
            if text_input:
                # Process text query
                response = agent.process_text_query(text_input, farm_id)
                return f"**Query:** {text_input}\n\n**Response:** {response}"
            
            elif audio:
                # Process audio query
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    agent.process_voice_query(audio, farm_id, self.session)
                )
                loop.close()
                
                output = "🎤 **Voice Query Results**\n\n"
                output += f"**Transcribed Text:** {result.transcribed_text}\n"
                output += f"**Detected Language:** {result.detected_language}\n"
                output += f"**Confidence:** {result.confidence_score:.1%}\n"
                output += f"**Intent:** {result.intent}\n"
                output += f"**Processing Time:** {result.processing_time_seconds:.2f}s\n\n"
                output += f"**Response:** {result.agent_response}"
                
                return output
            
            else:
                return "Please provide either text input or audio file"
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

def create_dashboard():
    """Create and configure the Gradio dashboard"""
    
    app = AgriBotXDashboard()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background-color: #f0f8f0;
    }
    """
    
    with gr.Blocks(css=css, title="AgriBotX Pro - Agricultural AI Assistant") as demo:
        
        gr.Markdown("""
        # 🌾 AgriBotX Pro - Agricultural AI Assistant
        
        Welcome to AgriBotX Pro, your comprehensive AI-powered agricultural assistant. 
        Get expert advice on crops, soil health, disease detection, irrigation, market prices, and more!
        """)
        
        with gr.Tabs():
            
            # Crop Recommendations Tab
            with gr.TabItem("🌱 Crop Recommendations"):
                gr.Markdown("Get AI-powered crop recommendations based on your farm conditions")
                
                with gr.Row():
                    with gr.Column():
                        crop_farm_id = gr.Textbox(label="Farm ID", placeholder="Enter your farm ID")
                        crop_type_filter = gr.Textbox(label="Crop Type (Optional)", placeholder="e.g., wheat, rice, tomato")
                        crop_recommend_btn = gr.Button("Get Recommendations", variant="primary")
                    
                    with gr.Column():
                        crop_output = gr.Markdown(label="Recommendations")
                
                crop_recommend_btn.click(
                    app.get_crop_recommendations,
                    inputs=[crop_farm_id, crop_type_filter],
                    outputs=crop_output
                )
            
            # Soil Analysis Tab
            with gr.TabItem("🌍 Soil Analysis"):
                gr.Markdown("Upload your soil report for comprehensive analysis and recommendations")
                
                with gr.Row():
                    with gr.Column():
                        soil_file = gr.File(label="Upload Soil Report", file_types=[".json", ".csv", ".xml", ".txt"])
                        soil_farm_id = gr.Textbox(label="Farm ID", placeholder="Enter your farm ID")
                        soil_analyze_btn = gr.Button("Analyze Soil", variant="primary")
                    
                    with gr.Column():
                        soil_output = gr.Markdown(label="Analysis Results")
                
                soil_analyze_btn.click(
                    app.analyze_soil_report,
                    inputs=[soil_file, soil_farm_id],
                    outputs=soil_output
                )
            
            # Disease Detection Tab
            with gr.TabItem("🦠 Disease Detection"):
                gr.Markdown("Upload plant images to detect diseases and get treatment recommendations")
                
                with gr.Row():
                    with gr.Column():
                        disease_image = gr.Image(label="Upload Plant Image", type="filepath")
                        disease_crop_type = gr.Dropdown(
                            choices=["tomato", "potato", "corn", "wheat", "rice", "cotton"],
                            label="Crop Type"
                        )
                        disease_detect_btn = gr.Button("Detect Disease", variant="primary")
                    
                    with gr.Column():
                        disease_output = gr.Markdown(label="Detection Results")
                
                disease_detect_btn.click(
                    app.detect_plant_disease,
                    inputs=[disease_image, disease_crop_type],
                    outputs=disease_output
                )
            
            # Irrigation Tab
            with gr.TabItem("💧 Irrigation"):
                gr.Markdown("Get smart irrigation scheduling based on weather and soil conditions")
                
                with gr.Row():
                    with gr.Column():
                        irr_farm_id = gr.Textbox(label="Farm ID", placeholder="Enter your farm ID")
                        irr_crop_type = gr.Dropdown(
                            choices=["wheat", "rice", "corn", "tomato", "potato", "cotton"],
                            label="Crop Type"
                        )
                        irr_growth_stage = gr.Dropdown(
                            choices=["initial", "development", "mid", "late"],
                            label="Growth Stage"
                        )
                        irr_schedule_btn = gr.Button("Get Schedule", variant="primary")
                    
                    with gr.Column():
                        irr_output = gr.Markdown(label="Irrigation Schedule")
                
                irr_schedule_btn.click(
                    app.get_irrigation_schedule,
                    inputs=[irr_farm_id, irr_crop_type, irr_growth_stage],
                    outputs=irr_output
                )
            
            # Market Prices Tab
            with gr.TabItem("📈 Market Prices"):
                gr.Markdown("Get current market prices and price forecasts for crops")
                
                with gr.Tabs():
                    with gr.TabItem("Current Prices"):
                        with gr.Row():
                            with gr.Column():
                                price_crop = gr.Textbox(label="Crop Name", placeholder="e.g., wheat, rice, tomato")
                                price_market = gr.Textbox(label="Market Name (Optional)", placeholder="e.g., delhi, mumbai")
                                price_get_btn = gr.Button("Get Prices", variant="primary")
                            
                            with gr.Column():
                                price_output = gr.Markdown(label="Current Prices")
                        
                        price_chart = gr.Plot(label="Price Chart")
                        
                        price_get_btn.click(
                            app.get_market_prices,
                            inputs=[price_crop, price_market],
                            outputs=[price_output, price_chart]
                        )
                    
                    with gr.TabItem("Price Forecast"):
                        with gr.Row():
                            with gr.Column():
                                forecast_crop = gr.Textbox(label="Crop Name", placeholder="e.g., wheat")
                                forecast_market = gr.Textbox(label="Market Name", placeholder="e.g., delhi")
                                forecast_days = gr.Slider(minimum=7, maximum=90, value=30, label="Forecast Days")
                                forecast_btn = gr.Button("Get Forecast", variant="primary")
                            
                            with gr.Column():
                                forecast_output = gr.Markdown(label="Price Forecast")
                        
                        forecast_btn.click(
                            app.get_price_forecast,
                            inputs=[forecast_crop, forecast_market, forecast_days],
                            outputs=forecast_output
                        )
            
            # Yield Prediction Tab
            with gr.TabItem("📊 Yield Prediction"):
                gr.Markdown("Predict crop yields using AI models and environmental data")
                
                with gr.Row():
                    with gr.Column():
                        yield_farm_id = gr.Textbox(label="Farm ID", placeholder="Enter your farm ID")
                        yield_crop_type = gr.Dropdown(
                            choices=["wheat", "rice", "corn", "tomato", "potato", "cotton"],
                            label="Crop Type"
                        )
                        yield_planting_date = gr.Textbox(label="Planting Date", placeholder="YYYY-MM-DD")
                        yield_predict_btn = gr.Button("Predict Yield", variant="primary")
                    
                    with gr.Column():
                        yield_output = gr.Markdown(label="Yield Prediction")
                
                yield_predict_btn.click(
                    app.predict_yield,
                    inputs=[yield_farm_id, yield_crop_type, yield_planting_date],
                    outputs=yield_output
                )
            
            # Voice Interface Tab
            with gr.TabItem("🎤 Voice Assistant"):
                gr.Markdown("Ask questions using voice or text - supports English and Hindi")
                
                with gr.Row():
                    with gr.Column():
                        voice_audio = gr.Audio(label="Record Voice Query", type="filepath")
                        voice_text = gr.Textbox(label="Or Type Your Question", placeholder="What crop should I grow?")
                        voice_farm_id = gr.Textbox(label="Farm ID (Optional)", placeholder="For personalized responses")
                        voice_query_btn = gr.Button("Process Query", variant="primary")
                    
                    with gr.Column():
                        voice_output = gr.Markdown(label="Response")
                
                voice_query_btn.click(
                    app.process_voice_query,
                    inputs=[voice_audio, voice_text, voice_farm_id],
                    outputs=voice_output
                )
                
                gr.Markdown("""
                **Supported Commands:**
                - "What crop should I grow?"
                - "My plant has a disease"
                - "When should I water?"
                - "What are the market prices?"
                - "What's the weather forecast?"
                - "How to test soil health?"
                """)
        
        gr.Markdown("""
        ---
        **AgriBotX Pro** - Empowering farmers with AI-driven agricultural insights
        
        For support, contact: support@agribotx.com
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )