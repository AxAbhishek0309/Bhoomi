"""
Soil Analyzer Agent
Parses soil reports and provides nutrient analysis and recommendations
"""

import json
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session
from db.models import SoilReport, Farm

logger = logging.getLogger(__name__)

@dataclass
class NutrientAnalysis:
    nutrient: str
    value: float
    unit: str
    status: str  # low, medium, high, optimal
    recommendation: str

@dataclass
class SoilAnalysisResult:
    ph_analysis: Dict[str, Any]
    nutrient_analysis: List[NutrientAnalysis]
    overall_health_score: float
    recommendations: List[str]
    deficiencies: List[str]
    amendments_needed: List[str]

class SoilAnalyzerAgent:
    """
    Intelligent soil analysis agent that processes soil reports and provides actionable insights
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nutrient_thresholds = config.get('nutrient_thresholds', {})
        self.supported_formats = config.get('supported_formats', ['json', 'csv', 'xml'])
        
        # Standard nutrient thresholds (can be overridden by config)
        self.default_thresholds = {
            'nitrogen': {'low': 0.02, 'medium': 0.05, 'high': 0.08, 'optimal': 0.06},
            'phosphorus': {'low': 10, 'medium': 25, 'high': 40, 'optimal': 30},
            'potassium': {'low': 100, 'medium': 200, 'high': 300, 'optimal': 250},
            'ph': {'acidic': 6.0, 'neutral': 7.0, 'alkaline': 8.0, 'optimal': 6.5},
            'organic_matter': {'low': 1.0, 'medium': 3.0, 'high': 5.0, 'optimal': 4.0}
        }
    
    async def analyze_soil_report(self, report_data: Union[str, Dict, pd.DataFrame], 
                                farm_id: str, session: Session) -> SoilAnalysisResult:
        """
        Analyze soil report data and provide comprehensive insights
        
        Args:
            report_data: Soil report in various formats (file path, dict, or DataFrame)
            farm_id: UUID of the farm
            session: Database session
            
        Returns:
            SoilAnalysisResult with detailed analysis
        """
        try:
            # Parse the soil report data
            parsed_data = await self._parse_soil_data(report_data)
            
            # Perform nutrient analysis
            nutrient_analysis = self._analyze_nutrients(parsed_data)
            
            # Analyze pH levels
            ph_analysis = self._analyze_ph(parsed_data)
            
            # Calculate overall health score
            health_score = self._calculate_health_score(parsed_data, nutrient_analysis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(parsed_data, nutrient_analysis, ph_analysis)
            
            # Identify deficiencies
            deficiencies = self._identify_deficiencies(nutrient_analysis)
            
            # Suggest amendments
            amendments = self._suggest_amendments(parsed_data, nutrient_analysis, ph_analysis)
            
            # Create analysis result
            result = SoilAnalysisResult(
                ph_analysis=ph_analysis,
                nutrient_analysis=nutrient_analysis,
                overall_health_score=health_score,
                recommendations=recommendations,
                deficiencies=deficiencies,
                amendments_needed=amendments
            )
            
            # Save to database
            await self._save_soil_report(farm_id, parsed_data, result, session)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing soil report: {e}")
            raise
    
    async def _parse_soil_data(self, report_data: Union[str, Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Parse soil data from various input formats"""
        
        if isinstance(report_data, dict):
            return report_data
        
        elif isinstance(report_data, pd.DataFrame):
            return report_data.to_dict('records')[0] if not report_data.empty else {}
        
        elif isinstance(report_data, str):
            # Assume it's a file path
            file_path = Path(report_data)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Soil report file not found: {report_data}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                return df.to_dict('records')[0] if not df.empty else {}
            
            elif file_extension == '.xml':
                # Basic XML parsing - would need more sophisticated parsing for real XML
                import xml.etree.ElementTree as ET
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                data = {}
                for child in root:
                    try:
                        data[child.tag] = float(child.text)
                    except (ValueError, TypeError):
                        data[child.tag] = child.text
                
                return data
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        else:
            raise ValueError(f"Unsupported data type: {type(report_data)}")
    
    def _analyze_nutrients(self, soil_data: Dict[str, Any]) -> List[NutrientAnalysis]:
        """Analyze nutrient levels in soil data"""
        nutrient_analysis = []
        
        # Map common field names to standard nutrient names
        field_mapping = {
            'nitrogen': ['nitrogen', 'n', 'n_percent', 'total_nitrogen'],
            'phosphorus': ['phosphorus', 'p', 'p_ppm', 'available_phosphorus'],
            'potassium': ['potassium', 'k', 'k_ppm', 'available_potassium'],
            'organic_matter': ['organic_matter', 'om', 'organic_carbon', 'oc']
        }
        
        for nutrient, possible_fields in field_mapping.items():
            value = None
            
            # Find the nutrient value in soil data
            for field in possible_fields:
                if field in soil_data:
                    try:
                        value = float(soil_data[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if value is not None:
                analysis = self._analyze_single_nutrient(nutrient, value)
                nutrient_analysis.append(analysis)
        
        return nutrient_analysis
    
    def _analyze_single_nutrient(self, nutrient: str, value: float) -> NutrientAnalysis:
        """Analyze a single nutrient level"""
        thresholds = self.nutrient_thresholds.get(nutrient, self.default_thresholds.get(nutrient, {}))
        
        # Determine status
        if value < thresholds.get('low', 0):
            status = 'low'
            recommendation = f"Apply {nutrient} fertilizer to increase levels"
        elif value > thresholds.get('high', float('inf')):
            status = 'high'
            recommendation = f"Reduce {nutrient} application to prevent toxicity"
        elif thresholds.get('low', 0) <= value <= thresholds.get('medium', float('inf')):
            status = 'medium'
            recommendation = f"Maintain current {nutrient} levels with regular application"
        else:
            status = 'optimal'
            recommendation = f"{nutrient.title()} levels are optimal"
        
        # Determine unit based on nutrient type
        unit = '%' if nutrient in ['nitrogen', 'organic_matter'] else 'ppm'
        
        return NutrientAnalysis(
            nutrient=nutrient,
            value=value,
            unit=unit,
            status=status,
            recommendation=recommendation
        )
    
    def _analyze_ph(self, soil_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze soil pH levels"""
        ph_fields = ['ph', 'ph_level', 'soil_ph', 'ph_value']
        ph_value = None
        
        for field in ph_fields:
            if field in soil_data:
                try:
                    ph_value = float(soil_data[field])
                    break
                except (ValueError, TypeError):
                    continue
        
        if ph_value is None:
            return {'status': 'unknown', 'recommendation': 'pH measurement needed'}
        
        # Classify pH level
        if ph_value < 6.0:
            status = 'acidic'
            recommendation = 'Apply lime to increase pH and reduce acidity'
            crop_suitability = 'Good for acid-loving crops like blueberries, potatoes'
        elif ph_value > 8.0:
            status = 'alkaline'
            recommendation = 'Apply sulfur or organic matter to reduce pH'
            crop_suitability = 'May limit nutrient availability for most crops'
        elif 6.0 <= ph_value <= 7.0:
            status = 'slightly acidic to neutral'
            recommendation = 'pH is in good range for most crops'
            crop_suitability = 'Suitable for most agricultural crops'
        else:  # 7.0 < pH <= 8.0
            status = 'slightly alkaline'
            recommendation = 'Monitor and consider slight acidification if needed'
            crop_suitability = 'Good for most crops, may affect iron availability'
        
        return {
            'value': ph_value,
            'status': status,
            'recommendation': recommendation,
            'crop_suitability': crop_suitability
        }
    
    def _calculate_health_score(self, soil_data: Dict[str, Any], 
                              nutrient_analysis: List[NutrientAnalysis]) -> float:
        """Calculate overall soil health score (0-100)"""
        scores = []
        
        # pH score
        ph_value = soil_data.get('ph', soil_data.get('ph_level', 7.0))
        try:
            ph_value = float(ph_value)
            if 6.0 <= ph_value <= 7.5:
                ph_score = 100
            elif 5.5 <= ph_value < 6.0 or 7.5 < ph_value <= 8.0:
                ph_score = 80
            elif 5.0 <= ph_value < 5.5 or 8.0 < ph_value <= 8.5:
                ph_score = 60
            else:
                ph_score = 40
            scores.append(ph_score)
        except (ValueError, TypeError):
            pass
        
        # Nutrient scores
        for analysis in nutrient_analysis:
            if analysis.status == 'optimal':
                scores.append(100)
            elif analysis.status == 'medium':
                scores.append(80)
            elif analysis.status == 'low':
                scores.append(50)
            else:  # high
                scores.append(60)
        
        # Organic matter score
        om_value = soil_data.get('organic_matter', soil_data.get('om', 0))
        try:
            om_value = float(om_value)
            if om_value >= 4.0:
                scores.append(100)
            elif om_value >= 3.0:
                scores.append(80)
            elif om_value >= 2.0:
                scores.append(60)
            else:
                scores.append(40)
        except (ValueError, TypeError):
            pass
        
        return sum(scores) / len(scores) if scores else 50.0
    
    def _generate_recommendations(self, soil_data: Dict[str, Any], 
                                nutrient_analysis: List[NutrientAnalysis],
                                ph_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive soil management recommendations"""
        recommendations = []
        
        # pH recommendations
        if ph_analysis.get('recommendation'):
            recommendations.append(ph_analysis['recommendation'])
        
        # Nutrient recommendations
        for analysis in nutrient_analysis:
            if analysis.status in ['low', 'high']:
                recommendations.append(analysis.recommendation)
        
        # General recommendations
        om_value = soil_data.get('organic_matter', 0)
        try:
            om_value = float(om_value)
            if om_value < 3.0:
                recommendations.append("Increase organic matter through compost or green manure")
        except (ValueError, TypeError):
            pass
        
        # Moisture recommendations
        moisture = soil_data.get('moisture_content', 0)
        try:
            moisture = float(moisture)
            if moisture < 10:
                recommendations.append("Improve water retention through organic matter addition")
            elif moisture > 30:
                recommendations.append("Improve drainage to prevent waterlogging")
        except (ValueError, TypeError):
            pass
        
        return recommendations
    
    def _identify_deficiencies(self, nutrient_analysis: List[NutrientAnalysis]) -> List[str]:
        """Identify nutrient deficiencies"""
        deficiencies = []
        
        for analysis in nutrient_analysis:
            if analysis.status == 'low':
                deficiencies.append(f"{analysis.nutrient.title()} deficiency")
        
        return deficiencies
    
    def _suggest_amendments(self, soil_data: Dict[str, Any], 
                          nutrient_analysis: List[NutrientAnalysis],
                          ph_analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific soil amendments"""
        amendments = []
        
        # pH amendments
        ph_value = ph_analysis.get('value', 7.0)
        if ph_value < 6.0:
            amendments.append("Agricultural lime (2-4 tons/hectare)")
        elif ph_value > 8.0:
            amendments.append("Elemental sulfur (200-500 kg/hectare)")
        
        # Nutrient amendments
        for analysis in nutrient_analysis:
            if analysis.status == 'low':
                if analysis.nutrient == 'nitrogen':
                    amendments.append("Urea or ammonium sulfate (100-200 kg/hectare)")
                elif analysis.nutrient == 'phosphorus':
                    amendments.append("Single super phosphate (200-300 kg/hectare)")
                elif analysis.nutrient == 'potassium':
                    amendments.append("Muriate of potash (100-150 kg/hectare)")
        
        # Organic matter amendments
        om_value = soil_data.get('organic_matter', 0)
        try:
            om_value = float(om_value)
            if om_value < 3.0:
                amendments.append("Compost or farmyard manure (5-10 tons/hectare)")
        except (ValueError, TypeError):
            pass
        
        return amendments
    
    async def _save_soil_report(self, farm_id: str, soil_data: Dict[str, Any], 
                              analysis_result: SoilAnalysisResult, session: Session):
        """Save soil report and analysis to database"""
        try:
            soil_report = SoilReport(
                farm_id=farm_id,
                ph_level=soil_data.get('ph', soil_data.get('ph_level')),
                nitrogen=soil_data.get('nitrogen', soil_data.get('n')),
                phosphorus=soil_data.get('phosphorus', soil_data.get('p')),
                potassium=soil_data.get('potassium', soil_data.get('k')),
                organic_matter=soil_data.get('organic_matter', soil_data.get('om')),
                moisture_content=soil_data.get('moisture_content'),
                report_date=datetime.utcnow(),
                lab_name=soil_data.get('lab_name', 'Unknown'),
                raw_data=soil_data
            )
            
            session.add(soil_report)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving soil report: {e}")
            session.rollback()
    
    def generate_soil_report_summary(self, analysis_result: SoilAnalysisResult) -> str:
        """Generate a human-readable summary of soil analysis"""
        summary = f"Soil Health Score: {analysis_result.overall_health_score:.1f}/100\n\n"
        
        # pH summary
        ph_info = analysis_result.ph_analysis
        summary += f"pH Level: {ph_info.get('value', 'N/A')} ({ph_info.get('status', 'Unknown')})\n"
        summary += f"pH Recommendation: {ph_info.get('recommendation', 'None')}\n\n"
        
        # Nutrient summary
        summary += "Nutrient Analysis:\n"
        for nutrient in analysis_result.nutrient_analysis:
            summary += f"- {nutrient.nutrient.title()}: {nutrient.value} {nutrient.unit} ({nutrient.status})\n"
        
        # Deficiencies
        if analysis_result.deficiencies:
            summary += f"\nDeficiencies Found: {', '.join(analysis_result.deficiencies)}\n"
        
        # Recommendations
        if analysis_result.recommendations:
            summary += "\nRecommendations:\n"
            for i, rec in enumerate(analysis_result.recommendations, 1):
                summary += f"{i}. {rec}\n"
        
        return summary