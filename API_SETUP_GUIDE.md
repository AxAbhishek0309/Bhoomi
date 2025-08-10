# 🔑 AgriBotX Pro API Setup Guide (Simplified)

This guide will help you get the essential API keys to make AgriBotX Pro work with **100% live, real data** using the simplest possible setup.

## 🚨 Required APIs (System won't work without this)

### 1. OpenAI API Key
**Purpose**: Powers the AI-based crop recommendations and natural language processing.

**How to get it**:
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Add to `.env`: `OPENAI_API_KEY=sk-your-key-here`

**Cost**: Pay-per-use, ~$0.002 per 1K tokens (very cheap for agricultural queries)

---

## 🌟 Recommended APIs (For best live data experience)

### 2. OpenWeatherMap API Key
**Purpose**: Real-time weather data and forecasts for irrigation and crop planning.

**How to get it**:
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for free account
3. Go to API Keys section
4. Copy your default API key
5. Add to `.env`: `OPENWEATHER_API_KEY=your-key-here`

**Cost**: FREE (up to 1,000 calls/day)

---

## 🆓 Free Data Sources (No API keys needed!)

These work automatically without any setup:

- **Open-Meteo**: Free weather data (no registration required)
- **NASA POWER**: Historical weather data (free)
- **NASA MODIS**: **REAL satellite NDVI data** (free, 250m resolution)
- **NASA EarthData**: **REAL satellite imagery** (free, multiple satellites)
- **USGS Landsat**: **REAL high-res satellite data** (free, 30m resolution)
- **Agmarknet Web Scraping**: Live crop prices (no API key needed!)

---

## 🚀 Super Quick Setup (2 minutes!)

1. **Copy the environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Add your OpenAI key to .env**:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   OPENWEATHER_API_KEY=your-weather-key-here  # Optional but recommended
   ```

3. **Test your setup**:
   ```bash
   python main.py --mode check-apis
   ```

4. **Start the system**:
   ```bash
   python main.py --mode web
   ```

---

## 🎯 What You Get With This Simple Setup

### With Just OpenAI Key:
- ✅ AI crop recommendations
- ✅ Free weather data (Open-Meteo, NASA)
- ✅ **REAL satellite NDVI data** (NASA MODIS, EarthData, Landsat)
- ✅ Live market prices (web scraping)
- ✅ Voice interface (text-based)

### With OpenAI + OpenWeatherMap:
- ✅ Everything above PLUS
- ✅ More accurate weather forecasts
- ✅ Better irrigation recommendations
- ✅ Higher data reliability

---

## 🔍 Check What's Working

Run this command anytime to see your API status:

```bash
python main.py --mode check-apis
```

You'll see:
- ✅ Which APIs are working
- ❌ Which APIs need setup
- 🆓 Which free sources are available
- ⏱️ Response times

---

## 💡 Why This Setup is Better

### No Complex Dependencies:
- ❌ No SentinelHub account needed
- ❌ No Twilio SMS setup required
- ❌ No complex API registrations
- ✅ Just 1-2 simple API keys

### Still Gets Live Data:
- ✅ Real weather from multiple sources
- ✅ **REAL satellite crop monitoring** (NASA MODIS, EarthData, Landsat)
- ✅ Current market prices via web scraping
- ✅ AI-powered recommendations

### Cost-Effective:
- 🆓 Most data sources are completely free
- 💰 OpenAI costs ~$1-5/month for typical usage
- 💰 OpenWeatherMap is free up to 1,000 calls/day

---

## 🆘 Troubleshooting

### "OpenAI API failed":
1. Check your API key is correct
2. Verify you have credits in your OpenAI account
3. Check the key starts with `sk-`

### "No weather data":
- Don't worry! The system uses free Open-Meteo automatically
- Add OpenWeatherMap key for better accuracy (optional)

### "No market prices":
- The system scrapes Agmarknet automatically (no key needed)
- Check your internet connection

### Getting started:
```bash
# Minimal setup - just OpenAI
echo "OPENAI_API_KEY=sk-your-key" > .env
python main.py --mode web

# Check what's working
python main.py --mode check-apis
```

---

## 📈 Upgrade Path

Start simple, add more later:

1. **Start**: OpenAI key only
2. **Add**: OpenWeatherMap for better weather
3. **Later**: Consider paid tiers if you need higher quotas

---

**Bottom Line**: You can get 90% of the functionality with just 1-2 free API keys! 🎉

The system is designed to work great with minimal setup while still providing real, live agricultural data.

Happy farming! 🌾🚜