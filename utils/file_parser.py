"""
File Parser Utilities
Handles parsing of various agricultural data file formats
"""

import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import csv
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SoilReportParser:
    """Parser for soil test reports in various formats"""
    
    @staticmethod
    def parse_soil_report(file_path: str) -> Dict[str, Any]:
        """
        Parse soil report from file
        
        Args:
            file_path: Path to the soil report file
            
        Returns:
            Dictionary containing parsed soil data
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.json':
                return SoilReportParser._parse_json_soil_report(file_path)
            elif file_extension == '.csv':
                return SoilReportParser._parse_csv_soil_report(file_path)
            elif file_extension == '.xml':
                return SoilReportParser._parse_xml_soil_report(file_path)
            elif file_extension == '.txt':
                return SoilReportParser._parse_text_soil_report(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error parsing soil report: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _parse_json_soil_report(file_path: Path) -> Dict[str, Any]:
        """Parse JSON soil report"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Standardize field names
            standardized_data = SoilReportParser._standardize_soil_fields(data)
            
            return {
                'source_format': 'json',
                'parsed_data': standardized_data,
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Error parsing JSON soil report: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _parse_csv_soil_report(file_path: Path) -> Dict[str, Any]:
        """Parse CSV soil report"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any encoding")
            
            # Convert to dictionary (assume first row contains the data)
            if len(df) > 0:
                data = df.iloc[0].to_dict()
            else:
                data = {}
            
            # Standardize field names
            standardized_data = SoilReportParser._standardize_soil_fields(data)
            
            return {
                'source_format': 'csv',
                'parsed_data': standardized_data,
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Error parsing CSV soil report: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _parse_xml_soil_report(file_path: Path) -> Dict[str, Any]:
        """Parse XML soil report"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            data = {}
            
            # Recursively extract data from XML
            def extract_xml_data(element, parent_key=''):
                for child in element:
                    key = child.tag
                    if parent_key:
                        key = f"{parent_key}_{key}"
                    
                    if len(child) > 0:
                        extract_xml_data(child, key)
                    else:
                        # Try to convert to appropriate type
                        value = child.text
                        if value:
                            try:
                                # Try float first
                                data[key] = float(value)
                            except ValueError:
                                try:
                                    # Try int
                                    data[key] = int(value)
                                except ValueError:
                                    # Keep as string
                                    data[key] = value.strip()
            
            extract_xml_data(root)
            
            # Standardize field names
            standardized_data = SoilReportParser._standardize_soil_fields(data)
            
            return {
                'source_format': 'xml',
                'parsed_data': standardized_data,
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Error parsing XML soil report: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _parse_text_soil_report(file_path: Path) -> Dict[str, Any]:
        """Parse text-based soil report using pattern matching"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {}
            
            # Common patterns for soil report data
            patterns = {
                'ph': [
                    r'ph[:\s]*(\d+\.?\d*)',
                    r'ph\s*level[:\s]*(\d+\.?\d*)',
                    r'soil\s*ph[:\s]*(\d+\.?\d*)'
                ],
                'nitrogen': [
                    r'nitrogen[:\s]*(\d+\.?\d*)',
                    r'n[:\s]*(\d+\.?\d*)\s*%',
                    r'total\s*nitrogen[:\s]*(\d+\.?\d*)'
                ],
                'phosphorus': [
                    r'phosphorus[:\s]*(\d+\.?\d*)',
                    r'p[:\s]*(\d+\.?\d*)\s*ppm',
                    r'available\s*phosphorus[:\s]*(\d+\.?\d*)'
                ],
                'potassium': [
                    r'potassium[:\s]*(\d+\.?\d*)',
                    r'k[:\s]*(\d+\.?\d*)\s*ppm',
                    r'available\s*potassium[:\s]*(\d+\.?\d*)'
                ],
                'organic_matter': [
                    r'organic\s*matter[:\s]*(\d+\.?\d*)',
                    r'om[:\s]*(\d+\.?\d*)\s*%',
                    r'organic\s*carbon[:\s]*(\d+\.?\d*)'
                ],
                'moisture_content': [
                    r'moisture[:\s]*(\d+\.?\d*)',
                    r'water\s*content[:\s]*(\d+\.?\d*)',
                    r'moisture\s*content[:\s]*(\d+\.?\d*)'
                ]
            }
            
            content_lower = content.lower()
            
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    match = re.search(pattern, content_lower)
                    if match:
                        try:
                            data[field] = float(match.group(1))
                            break
                        except ValueError:
                            continue
            
            # Extract lab information
            lab_patterns = [
                r'lab[:\s]*([^\n]+)',
                r'laboratory[:\s]*([^\n]+)',
                r'tested\s*by[:\s]*([^\n]+)'
            ]
            
            for pattern in lab_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    data['lab_name'] = match.group(1).strip()
                    break
            
            # Extract date
            date_patterns = [
                r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'tested\s*on[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'report\s*date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content)
                if match:
                    data['report_date'] = match.group(1)
                    break
            
            # Standardize field names
            standardized_data = SoilReportParser._standardize_soil_fields(data)
            
            return {
                'source_format': 'text',
                'parsed_data': standardized_data,
                'raw_data': data,
                'raw_content': content
            }
            
        except Exception as e:
            logger.error(f"Error parsing text soil report: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _standardize_soil_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize soil data field names and values"""
        try:
            standardized = {}
            
            # Field name mappings
            field_mappings = {
                'ph': ['ph', 'ph_level', 'soil_ph', 'ph_value'],
                'nitrogen': ['nitrogen', 'n', 'n_percent', 'total_nitrogen', 'n_content'],
                'phosphorus': ['phosphorus', 'p', 'p_ppm', 'available_phosphorus', 'p_content'],
                'potassium': ['potassium', 'k', 'k_ppm', 'available_potassium', 'k_content'],
                'organic_matter': ['organic_matter', 'om', 'organic_carbon', 'oc', 'om_percent'],
                'moisture_content': ['moisture_content', 'moisture', 'water_content', 'mc'],
                'lab_name': ['lab_name', 'laboratory', 'lab', 'tested_by'],
                'report_date': ['report_date', 'date', 'test_date', 'sample_date']
            }
            
            # Standardize field names
            for standard_name, possible_names in field_mappings.items():
                for possible_name in possible_names:
                    if possible_name in data:
                        value = data[possible_name]
                        
                        # Convert to appropriate type
                        if standard_name in ['ph', 'nitrogen', 'phosphorus', 'potassium', 
                                           'organic_matter', 'moisture_content']:
                            try:
                                standardized[standard_name] = float(value)
                            except (ValueError, TypeError):
                                pass
                        else:
                            standardized[standard_name] = str(value)
                        break
            
            # Validate and adjust values
            if 'ph' in standardized:
                ph = standardized['ph']
                if ph < 0 or ph > 14:
                    logger.warning(f"Invalid pH value: {ph}")
                    if ph > 14:
                        standardized['ph'] = ph / 10  # Might be in different scale
            
            if 'nitrogen' in standardized:
                n = standardized['nitrogen']
                if n > 1:  # Likely in percentage, convert to decimal
                    standardized['nitrogen'] = n / 100
            
            # Add metadata
            standardized['parsed_at'] = datetime.now().isoformat()
            standardized['data_quality'] = SoilReportParser._assess_data_quality(standardized)
            
            return standardized
            
        except Exception as e:
            logger.error(f"Error standardizing soil fields: {e}")
            return data
    
    @staticmethod
    def _assess_data_quality(data: Dict[str, Any]) -> str:
        """Assess the quality of parsed soil data"""
        try:
            required_fields = ['ph', 'nitrogen', 'phosphorus', 'potassium']
            present_fields = sum(1 for field in required_fields if field in data)
            
            if present_fields == len(required_fields):
                return 'complete'
            elif present_fields >= len(required_fields) * 0.75:
                return 'good'
            elif present_fields >= len(required_fields) * 0.5:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 'unknown'

class CropDataParser:
    """Parser for crop-related data files"""
    
    @staticmethod
    def parse_crop_prices(file_path: str) -> List[Dict[str, Any]]:
        """Parse crop price data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'crop': 'crop_name',
                'commodity': 'crop_name',
                'market': 'market_name',
                'mandi': 'market_name',
                'min': 'min_price',
                'max': 'max_price',
                'modal': 'modal_price',
                'average': 'modal_price',
                'date': 'price_date'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Convert to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error parsing crop prices: {e}")
            return []
    
    @staticmethod
    def parse_weather_data(file_path: str) -> List[Dict[str, Any]]:
        """Parse weather data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'temp': 'temperature',
                'temperature_c': 'temperature',
                'humidity': 'humidity_percent',
                'rain': 'rainfall_mm',
                'rainfall': 'rainfall_mm',
                'wind': 'wind_speed_kmh',
                'pressure': 'pressure_hpa',
                'date': 'recorded_at'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Convert date column
            if 'recorded_at' in df.columns:
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return []

def parse_soil_report(file_path: str) -> Dict[str, Any]:
    """Convenience function to parse soil reports"""
    return SoilReportParser.parse_soil_report(file_path)

def validate_file_format(file_path: str, expected_format: str) -> bool:
    """Validate if file matches expected format"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        file_extension = file_path.suffix.lower()
        expected_extension = f".{expected_format.lower()}"
        
        return file_extension == expected_extension
        
    except Exception as e:
        logger.error(f"Error validating file format: {e}")
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get information about a file"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found'}
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'extension': file_path.suffix,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'is_readable': os.access(file_path, os.R_OK)
        }
        
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {'error': str(e)}