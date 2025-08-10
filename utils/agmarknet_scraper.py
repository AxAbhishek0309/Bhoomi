"""
Agmarknet Web Scraper
Scrapes live crop prices from Agmarknet website without requiring API keys
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
import time
import random

logger = logging.getLogger(__name__)

class AgmarknetScraper:
    """
    Web scraper for Agmarknet crop prices - no API key required
    """
    
    def __init__(self):
        self.base_url = "https://agmarknet.gov.in"
        self.session = requests.Session()
        
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Crop name mappings for better search
        self.crop_mappings = {
            'wheat': ['wheat', 'gehun'],
            'rice': ['rice', 'paddy', 'chawal'],
            'tomato': ['tomato', 'tamatar'],
            'onion': ['onion', 'pyaz'],
            'potato': ['potato', 'aloo'],
            'cotton': ['cotton', 'kapas'],
            'sugarcane': ['sugarcane', 'ganna'],
            'soybean': ['soybean', 'soya'],
            'corn': ['maize', 'corn', 'makka'],
            'turmeric': ['turmeric', 'haldi']
        }
    
    async def get_crop_prices(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape current crop prices from Agmarknet
        
        Args:
            crop_name: Name of the crop
            markets: List of markets (optional)
            
        Returns:
            List of price data dictionaries
        """
        try:
            logger.info(f"Scraping Agmarknet prices for {crop_name}")
            
            # Get crop variations
            crop_variations = self.crop_mappings.get(crop_name.lower(), [crop_name])
            
            all_prices = []
            
            for crop_variant in crop_variations:
                prices = await self._scrape_crop_variant(crop_variant, markets)
                all_prices.extend(prices)
                
                # Add delay to be respectful to the server
                time.sleep(random.uniform(1, 3))
            
            # Remove duplicates and return
            unique_prices = self._remove_duplicates(all_prices)
            
            logger.info(f"Found {len(unique_prices)} price records for {crop_name}")
            return unique_prices
            
        except Exception as e:
            logger.error(f"Error scraping Agmarknet for {crop_name}: {e}")
            return []
    
    async def _scrape_crop_variant(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape prices for a specific crop variant"""
        try:
            # Method 1: Try the price search page
            prices = await self._scrape_price_search(crop_name, markets)
            if prices:
                return prices
            
            # Method 2: Try the daily price report
            prices = await self._scrape_daily_report(crop_name, markets)
            if prices:
                return prices
            
            # Method 3: Try the commodity-wise report
            prices = await self._scrape_commodity_report(crop_name, markets)
            return prices
            
        except Exception as e:
            logger.error(f"Error scraping crop variant {crop_name}: {e}")
            return []
    
    async def _scrape_price_search(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape from the price search functionality"""
        try:
            # Agmarknet price search URL
            search_url = f"{self.base_url}/SearchCmmMkt.aspx"
            
            # Get the search page first
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for price tables
            price_tables = soup.find_all('table', {'class': ['table', 'price-table', 'data-table']})
            
            prices = []
            for table in price_tables:
                table_prices = self._parse_price_table(table, crop_name)
                prices.extend(table_prices)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error in price search scraping: {e}")
            return []
    
    async def _scrape_daily_report(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape from daily price reports"""
        try:
            # Try to access daily price report
            daily_url = f"{self.base_url}/PriceAndArrivals/DateWisePriceAndArrivals.aspx"
            
            response = self.session.get(daily_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for data tables
            tables = soup.find_all('table')
            
            prices = []
            for table in tables:
                table_prices = self._parse_price_table(table, crop_name)
                prices.extend(table_prices)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error in daily report scraping: {e}")
            return []
    
    async def _scrape_commodity_report(self, crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape from commodity-wise reports"""
        try:
            # Alternative approach: scrape from commodity pages
            commodity_url = f"{self.base_url}/PriceAndArrivals/CommodityWiseDaily.aspx"
            
            response = self.session.get(commodity_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for commodity data
            tables = soup.find_all('table')
            
            prices = []
            for table in tables:
                table_prices = self._parse_price_table(table, crop_name)
                prices.extend(table_prices)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error in commodity report scraping: {e}")
            return []
    
    def _parse_price_table(self, table, crop_name: str) -> List[Dict[str, Any]]:
        """Parse price data from HTML table"""
        try:
            prices = []
            
            if not table:
                return prices
            
            # Get table rows
            rows = table.find_all('tr')
            
            if len(rows) < 2:  # Need at least header + 1 data row
                return prices
            
            # Try to identify header row
            header_row = rows[0]
            headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]
            
            # Map common column names
            col_mapping = {
                'commodity': ['commodity', 'crop', 'item'],
                'market': ['market', 'mandi', 'place'],
                'state': ['state', 'district'],
                'min_price': ['min', 'minimum', 'min_price', 'min price'],
                'max_price': ['max', 'maximum', 'max_price', 'max price'],
                'modal_price': ['modal', 'average', 'modal_price', 'modal price', 'avg'],
                'date': ['date', 'price_date', 'arrival_date']
            }
            
            # Find column indices
            col_indices = {}
            for key, variations in col_mapping.items():
                for i, header in enumerate(headers):
                    if any(var in header for var in variations):
                        col_indices[key] = i
                        break
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 3:  # Need at least 3 columns
                    continue
                
                cell_texts = [cell.get_text().strip() for cell in cells]
                
                # Check if this row contains our crop
                commodity_text = ""
                if 'commodity' in col_indices:
                    commodity_text = cell_texts[col_indices['commodity']].lower()
                else:
                    # Check first few columns for crop name
                    commodity_text = ' '.join(cell_texts[:3]).lower()
                
                if not any(crop_var in commodity_text for crop_var in [crop_name.lower()]):
                    continue
                
                # Extract price data
                try:
                    price_data = {
                        'crop_name': crop_name,
                        'market_name': cell_texts[col_indices.get('market', 1)] if 'market' in col_indices else 'Unknown',
                        'state': cell_texts[col_indices.get('state', 2)] if 'state' in col_indices else 'Unknown',
                        'min_price': self._extract_price(cell_texts[col_indices.get('min_price', 3)] if 'min_price' in col_indices else '0'),
                        'max_price': self._extract_price(cell_texts[col_indices.get('max_price', 4)] if 'max_price' in col_indices else '0'),
                        'modal_price': self._extract_price(cell_texts[col_indices.get('modal_price', 5)] if 'modal_price' in col_indices else '0'),
                        'price_date': datetime.now(),  # Default to today
                        'unit': 'quintal',
                        'source': 'agmarknet_scraper'
                    }
                    
                    # Only add if we have valid price data
                    if price_data['modal_price'] > 0:
                        prices.append(price_data)
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing row: {e}")
                    continue
            
            return prices
            
        except Exception as e:
            logger.error(f"Error parsing price table: {e}")
            return []
    
    def _extract_price(self, price_text: str) -> float:
        """Extract numeric price from text"""
        try:
            # Remove non-numeric characters except decimal point
            price_clean = re.sub(r'[^\d.]', '', price_text)
            
            if not price_clean:
                return 0.0
            
            return float(price_clean)
            
        except (ValueError, TypeError):
            return 0.0
    
    def _remove_duplicates(self, prices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate price entries"""
        try:
            seen = set()
            unique_prices = []
            
            for price in prices:
                # Create a key for deduplication
                key = (
                    price['crop_name'],
                    price['market_name'],
                    price['state'],
                    price['modal_price']
                )
                
                if key not in seen:
                    seen.add(key)
                    unique_prices.append(price)
            
            return unique_prices
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return prices
    
    async def get_market_trends(self, crop_name: str, days: int = 30) -> Dict[str, Any]:
        """Get market trends for a crop (simplified version)"""
        try:
            # Get current prices
            current_prices = await self.get_crop_prices(crop_name)
            
            if not current_prices:
                return {}
            
            # Calculate basic statistics
            modal_prices = [p['modal_price'] for p in current_prices if p['modal_price'] > 0]
            
            if not modal_prices:
                return {}
            
            trends = {
                'crop_name': crop_name,
                'average_price': sum(modal_prices) / len(modal_prices),
                'min_price': min(modal_prices),
                'max_price': max(modal_prices),
                'price_range': max(modal_prices) - min(modal_prices),
                'markets_count': len(set(p['market_name'] for p in current_prices)),
                'data_points': len(current_prices),
                'last_updated': datetime.now().isoformat()
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """Test if Agmarknet website is accessible"""
        try:
            response = self.session.get(self.base_url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Agmarknet connection test failed: {e}")
            return False

# Convenience function for easy import
async def scrape_agmarknet_prices(crop_name: str, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convenience function to scrape Agmarknet prices"""
    scraper = AgmarknetScraper()
    return await scraper.get_crop_prices(crop_name, markets)

if __name__ == "__main__":
    import asyncio
    
    async def test_scraper():
        scraper = AgmarknetScraper()
        
        # Test connection
        if scraper.test_connection():
            print("✅ Agmarknet connection successful")
        else:
            print("❌ Agmarknet connection failed")
            return
        
        # Test scraping
        crops = ['wheat', 'rice', 'tomato']
        
        for crop in crops:
            print(f"\n🌾 Scraping prices for {crop}...")
            prices = await scraper.get_crop_prices(crop)
            
            if prices:
                print(f"✅ Found {len(prices)} price records")
                for price in prices[:3]:  # Show first 3
                    print(f"   {price['market_name']}: ₹{price['modal_price']}/quintal")
            else:
                print(f"❌ No prices found for {crop}")
    
    asyncio.run(test_scraper())