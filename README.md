# AgriBotX Pro 🌾🤖

A modular, multi-agent AI system for precision agriculture that helps farmers and agronomists make data-driven decisions using real-world agricultural data.

## Features

- **Multi-Agent Architecture**: Specialized AI agents for different agricultural tasks
- **Real-time Data Integration**: Weather APIs, satellite imagery, soil reports, market prices
- **Voice Interface**: Rural-friendly voice commands using Whisper
- **Disease Detection**: Vision Transformers for crop disease identification
- **Market Intelligence**: Live crop price tracking and forecasting
- **Spatial Analytics**: PostGIS-enabled geospatial queries
- **Modular Design**: Easy to extend and contribute to

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/agribotx-pro.git
cd agribotx_pro

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python db/init_db.py

# Run CLI
python cli/agribotx_cli.py

# Or start web UI
python ui/dashboard.py
```

## Architecture

### Core Agents
- **CropRecommenderAgent**: LLM-powered crop selection based on soil, weather, market data
- **SoilAnalyzerAgent**: Parses soil reports and identifies nutrient deficiencies
- **DiseaseDetectorAgent**: Vision Transformer-based plant disease detection
- **IrrigationAdvisorAgent**: Smart irrigation scheduling using weather forecasts
- **MarketForecastAgent**: Crop price trends and profitability analysis
- **YieldPredictorAgent**: ML-based yield prediction using NDVI and environmental data
- **VoiceInterfaceAgent**: Speech-to-text processing for rural accessibility

### Data Sources
- **Weather**: Open-Meteo, NASA POWER APIs
- **Satellite**: SentinelHub NDVI data
- **Market**: Agmarknet API with CSV fallbacks
- **Soil**: JSON/CSV report parsing
- **Images**: Crop disease image classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new agents
4. Submit a pull request

## License

MIT License - see LICENSE file for details