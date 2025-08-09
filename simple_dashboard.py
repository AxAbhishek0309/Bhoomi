#!/usr/bin/env python3
"""
AgriBotX Pro - Simplified Local Dashboard
A simplified version for local testing without complex dependencies
"""

import gradio as gr
import pandas as pd
import json
import os
from datetime import datetime
import sqlite3
from pathlib import Path

class SimplifiedAgriBotX:
    """Simplified version of AgriBotX for local testing"""
    
    def __init__(self):
        self.setup_database()
        self.load_sample_data()
    
    def setup_database(self):
        """Setup SQLite database for local testing"""
        self.db_path = "agribotx_local.db"
        conn = sqlite3.connect(self.db_path)
        
        # Create simple tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS farms (
                id TEXT PRIMARY KEY,
                name TEXT,
                owner_name TEXT,
                latitude REAL,
                longitude REAL,
                area_hectares REAL,
                soil_type TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_name TEXT,
                market_name TEXT,
                min_price REAL,
                max_price REAL,
                modal_price REAL,
                price_date TEXT
            )
        """)
        
        # Insert sample data
        conn.execute("""
            INSERT OR REPLACE INTO farms VALUES 
            ('farm_001', 'Green Valley Farm', 'Rajesh Kumar', 28.6139, 77.2090, 5.5, 'Loamy'),
            ('farm_002', 'Sunrise Agriculture', 'Priya Sharma', 28.5355, 77.3910, 8.2, 'Clay')
        """)
        
        conn.commit()
        conn.close()
    
    def load_sample_data(self):
        """Load sample market data"""
        try:
            self.market_data = pd.read_csv('data/fallback_prices.csv')
        except:
            # Create sample data if file doesn't exist
            self.market_data = pd.DataFrame({
                'crop_name': ['wheat', 'rice', 'tomato', 'onion', 'potato'] * 5,
                'market_name': ['Delhi Mandi'] * 25,
                'min_price': [1800, 2200, 800, 1200, 600] * 5,
                'max_price': [2200, 2800, 1200, 1600, 900] * 5,
                'modal_price': [2000, 2500, 1000, 1400, 750] * 5,
                'price_date': ['2024-01-15'] * 25
            })
    
    def get_crop_recommendations(self, farm_id: str) -> str:
        """Get crop recommendations for a farm"""
        if not farm_id:
            return "Please provide a farm ID"
        
        # Simulate crop recommendations
        recommendations = {
            'farm_001': {
                'crops': [
                    {'name': 'Wheat', 'suitability': 85, 'yield': '4.2 tons/hectare', 'market': 'High'},
                    {'name': 'Rice', 'suitability': 78, 'yield': '5.8 tons/hectare', 'market': 'Medium'},
                    {'name': 'Tomato', 'suitability': 92, 'yield': '45 tons/hectare', 'market': 'High'}
                ],
                'advice': 'Your loamy soil is excellent for tomato cultivation. Consider drip irrigation for better water management.'
            },
            'farm_002': {
                'crops': [
                    {'name': 'Rice', 'suitability': 88, 'yield': '6.2 tons/hectare', 'market': 'High'},
                    {'name': 'Cotton', 'suitability': 82, 'yield': '2.1 tons/hectare', 'market': 'Medium'},
                    {'name': 'Sugarcane', 'suitability': 75, 'yield': '65 tons/hectare', 'market': 'Medium'}
                ],
                'advice': 'Clay soil retains water well, making it suitable for rice cultivation. Monitor drainage during monsoon.'
            }
        }
        
        farm_data = recommendations.get(farm_id, recommendations['farm_001'])
        
        output = "🌱 **Crop Recommendations**\n\n"
        
        for i, crop in enumerate(farm_data['crops'], 1):
            output += f"**{i}. {crop['name']}**\n"
            output += f"   - Suitability Score: {crop['suitability']}/100\n"
            output += f"   - Expected Yield: {crop['yield']}\n"
            output += f"   - Market Potential: {crop['market']}\n\n"
        
        output += f"**General Advice:**\n{farm_data['advice']}"
        
        return output
    
    def analyze_soil_report(self, file, farm_id: str) -> str:
        """Analyze uploaded soil report"""
        if not file:
            return "Please upload a soil report file"
        
        if not farm_id:
            return "Please provide a farm ID"
        
        try:
            # Try to read the uploaded file
            if file.name.endswith('.json'):
                with open(file.name, 'r') as f:
                    soil_data = json.load(f)
            elif file.name.endswith('.csv'):
                soil_df = pd.read_csv(file.name)
                soil_data = soil_df.to_dict('records')[0] if not soil_df.empty else {}
            else:
                return "Unsupported file format. Please upload JSON or CSV files."
            
            # Simulate soil analysis
            output = "🌍 **Soil Analysis Results**\n\n"
            output += f"**Overall Health Score: 78.5/100**\n\n"
            
            # Extract key parameters
            ph = soil_data.get('ph_level', soil_data.get('ph', 6.8))
            nitrogen = soil_data.get('nitrogen', 0.045)
            phosphorus = soil_data.get('phosphorus', 22.5)
            potassium = soil_data.get('potassium', 180.0)
            
            output += f"**pH Analysis:**\n"
            output += f"- Level: {ph} (Optimal)\n"
            output += f"- Recommendation: pH is in good range for most crops\n\n"
            
            output += "**Nutrient Analysis:**\n"
            output += f"- Nitrogen: {nitrogen}% (Good)\n"
            output += f"- Phosphorus: {phosphorus} ppm (Adequate)\n"
            output += f"- Potassium: {potassium} ppm (Good)\n\n"
            
            output += "**Recommendations:**\n"
            output += "• Maintain current nutrient levels with regular monitoring\n"
            output += "• Consider organic compost to improve soil structure\n"
            output += "• Apply balanced fertilizer during growing season\n"
            
            return output
            
        except Exception as e:
            return f"Error analyzing soil report: {str(e)}"
    
    def detect_plant_disease(self, image, crop_type: str) -> str:
        """Simulate disease detection"""
        if not image:
            return "Please upload a plant image"
        
        if not crop_type:
            return "Please specify the crop type"
        
        # Simulate disease detection results
        diseases = {
            'tomato': {
                'disease': 'Early Blight',
                'confidence': 0.87,
                'severity': 'Medium',
                'treatment': [
                    'Apply copper-based fungicide',
                    'Remove affected leaves',
                    'Improve air circulation',
                    'Avoid overhead watering'
                ]
            },
            'potato': {
                'disease': 'Late Blight',
                'confidence': 0.92,
                'severity': 'High',
                'treatment': [
                    'Apply systemic fungicide immediately',
                    'Remove infected plants',
                    'Improve drainage',
                    'Monitor weather conditions'
                ]
            }
        }
        
        disease_data = diseases.get(crop_type, diseases['tomato'])
        
        output = "🦠 **Disease Detection Results**\n\n"
        output += f"**Detected Disease:** {disease_data['disease']}\n"
        output += f"**Confidence:** {disease_data['confidence']:.1%}\n"
        output += f"**Severity:** {disease_data['severity']}\n\n"
        
        output += "**Treatment Recommendations:**\n"
        for treatment in disease_data['treatment']:
            output += f"• {treatment}\n"
        
        return output
    
    def get_market_prices(self, crop_name: str) -> tuple:
        """Get market prices and create simple visualization"""
        if not crop_name:
            return "Please provide a crop name", None
        
        # Filter data for the specified crop
        crop_data = self.market_data[self.market_data['crop_name'].str.lower() == crop_name.lower()]
        
        if crop_data.empty:
            return f"No price data available for {crop_name}", None
        
        output = f"📈 **Current Prices for {crop_name.title()}**\n\n"
        
        for _, row in crop_data.head(5).iterrows():
            output += f"**{row['market_name']}**\n"
            output += f"- Min Price: ₹{row['min_price']:.0f}/quintal\n"
            output += f"- Max Price: ₹{row['max_price']:.0f}/quintal\n"
            output += f"- Modal Price: ₹{row['modal_price']:.0f}/quintal\n\n"
        
        # Create simple price chart data
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Min Price',
                x=crop_data['market_name'].head(5),
                y=crop_data['min_price'].head(5),
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Modal Price',
                x=crop_data['market_name'].head(5),
                y=crop_data['modal_price'].head(5),
                marker_color='blue'
            ))
            fig.add_trace(go.Bar(
                name='Max Price',
                x=crop_data['market_name'].head(5),
                y=crop_data['max_price'].head(5),
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title=f"Current Prices for {crop_name.title()}",
                xaxis_title="Market",
                yaxis_title="Price (₹/quintal)",
                barmode='group'
            )
            
            return output, fig
        except:
            return output, None
    
    def get_irrigation_schedule(self, farm_id: str, crop_type: str, growth_stage: str) -> str:
        """Get irrigation schedule"""
        if not all([farm_id, crop_type, growth_stage]):
            return "Please provide farm ID, crop type, and growth stage"
        
        # Simulate irrigation recommendations
        schedules = {
            'wheat': {
                'initial': {'water': 800, 'frequency': 'Every 7 days', 'method': 'Flood'},
                'development': {'water': 1200, 'frequency': 'Every 5 days', 'method': 'Sprinkler'},
                'mid': {'water': 1500, 'frequency': 'Every 4 days', 'method': 'Drip'},
                'late': {'water': 600, 'frequency': 'Every 10 days', 'method': 'Drip'}
            },
            'rice': {
                'initial': {'water': 2000, 'frequency': 'Continuous flooding', 'method': 'Flood'},
                'development': {'water': 2500, 'frequency': 'Maintain water level', 'method': 'Flood'},
                'mid': {'water': 2200, 'frequency': 'Maintain water level', 'method': 'Flood'},
                'late': {'water': 800, 'frequency': 'Intermittent', 'method': 'Controlled'}
            }
        }
        
        crop_schedule = schedules.get(crop_type, schedules['wheat'])
        stage_data = crop_schedule.get(growth_stage, crop_schedule['initial'])
        
        output = "💧 **Irrigation Schedule**\n\n"
        output += f"**Crop:** {crop_type.title()}\n"
        output += f"**Growth Stage:** {growth_stage.title()}\n"
        output += f"**Water Amount:** {stage_data['water']} liters per application\n"
        output += f"**Frequency:** {stage_data['frequency']}\n"
        output += f"**Method:** {stage_data['method']}\n\n"
        
        output += "**Weather Considerations:**\n"
        output += "• Check weather forecast before irrigation\n"
        output += "• Reduce frequency during rainy season\n"
        output += "• Monitor soil moisture levels\n"
        
        return output

def create_simple_dashboard():
    """Create simplified Gradio dashboard"""
    
    app = SimplifiedAgriBotX()
    
    with gr.Blocks(title="AgriBotX Pro - Agricultural AI Assistant") as demo:
        
        gr.Markdown("""
        # 🌾 AgriBotX Pro - Agricultural AI Assistant (Local Demo)
        
        Welcome to AgriBotX Pro! This is a simplified local demo version.
        Get AI-powered agricultural advice for your farming needs.
        """)
        
        with gr.Tabs():
            
            # Crop Recommendations Tab
            with gr.TabItem("🌱 Crop Recommendations"):
                gr.Markdown("Get AI-powered crop recommendations based on your farm conditions")
                
                with gr.Row():
                    with gr.Column():
                        crop_farm_id = gr.Textbox(
                            label="Farm ID", 
                            placeholder="Enter farm_001 or farm_002",
                            value="farm_001"
                        )
                        crop_recommend_btn = gr.Button("Get Recommendations", variant="primary")
                    
                    with gr.Column():
                        crop_output = gr.Markdown(label="Recommendations")
                
                crop_recommend_btn.click(
                    app.get_crop_recommendations,
                    inputs=[crop_farm_id],
                    outputs=crop_output
                )
            
            # Soil Analysis Tab
            with gr.TabItem("🌍 Soil Analysis"):
                gr.Markdown("Upload your soil report for analysis (try the sample files in data/mock_soil_reports/)")
                
                with gr.Row():
                    with gr.Column():
                        soil_file = gr.File(label="Upload Soil Report", file_types=[".json", ".csv"])
                        soil_farm_id = gr.Textbox(
                            label="Farm ID", 
                            placeholder="Enter farm_001 or farm_002",
                            value="farm_001"
                        )
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
                gr.Markdown("Upload plant images to detect diseases (demo mode - simulated results)")
                
                with gr.Row():
                    with gr.Column():
                        disease_image = gr.Image(label="Upload Plant Image", type="filepath")
                        disease_crop_type = gr.Dropdown(
                            choices=["tomato", "potato", "corn", "wheat", "rice"],
                            label="Crop Type",
                            value="tomato"
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
                gr.Markdown("Get smart irrigation scheduling recommendations")
                
                with gr.Row():
                    with gr.Column():
                        irr_farm_id = gr.Textbox(
                            label="Farm ID", 
                            placeholder="Enter farm_001 or farm_002",
                            value="farm_001"
                        )
                        irr_crop_type = gr.Dropdown(
                            choices=["wheat", "rice", "corn", "tomato", "potato"],
                            label="Crop Type",
                            value="wheat"
                        )
                        irr_growth_stage = gr.Dropdown(
                            choices=["initial", "development", "mid", "late"],
                            label="Growth Stage",
                            value="development"
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
                gr.Markdown("Get current market prices for crops")
                
                with gr.Row():
                    with gr.Column():
                        price_crop = gr.Dropdown(
                            choices=["wheat", "rice", "tomato", "onion", "potato", "cotton"],
                            label="Crop Name",
                            value="wheat"
                        )
                        price_get_btn = gr.Button("Get Prices", variant="primary")
                    
                    with gr.Column():
                        price_output = gr.Markdown(label="Current Prices")
                
                price_chart = gr.Plot(label="Price Chart")
                
                price_get_btn.click(
                    app.get_market_prices,
                    inputs=[price_crop],
                    outputs=[price_output, price_chart]
                )
        
        gr.Markdown("""
        ---
        **AgriBotX Pro Local Demo** - Simplified version for local testing
        
        **Sample Farm IDs:** farm_001, farm_002
        
        **Note:** This is a demo version with simulated data. The full version includes:
        - Real API integrations (Weather, Satellite, Market data)
        - Advanced ML models for disease detection and yield prediction
        - Voice interface with speech recognition
        - Database integration with spatial queries
        - Production-ready deployment with Docker
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_simple_dashboard()
    print("🚀 Starting AgriBotX Pro Local Demo...")
    print("📱 The web interface will open in your browser")
    print("🌐 Access URL: http://localhost:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )