"""
Voice Interface Agent
Handles speech-to-text processing and routes voice queries to appropriate agents
"""

import os
import whisper
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile
import wave
import json
import re

from sqlalchemy.orm import Session
from db.models import VoiceQuery, Farm

logger = logging.getLogger(__name__)

@dataclass
class VoiceQueryResult:
    transcribed_text: str
    detected_language: str
    confidence_score: float
    intent: str
    entities: Dict[str, Any]
    agent_response: str
    processing_time_seconds: float

@dataclass
class QueryIntent:
    intent_type: str  # crop_recommendation, disease_detection, irrigation, market_info, etc.
    confidence: float
    entities: Dict[str, Any]  # extracted entities like crop names, locations, etc.

class VoiceInterfaceAgent:
    """
    AI-powered voice interface for rural farmers using Whisper and NLP
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_model_name = config.get('whisper_model', 'base')
        self.supported_languages = config.get('supported_languages', ['en', 'hi'])
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.max_audio_duration = config.get('max_audio_duration', 60)
        
        # Load Whisper model
        self.whisper_model = None
        self._load_whisper_model()
        
        # Intent patterns for different queries
        self.intent_patterns = {
            'crop_recommendation': [
                r'what.*crop.*grow',
                r'which.*crop.*plant',
                r'recommend.*crop',
                r'best.*crop.*season',
                r'कौन सी फसल.*उगाएं',  # Hindi
                r'फसल.*सुझाव'
            ],
            'disease_detection': [
                r'disease.*plant',
                r'what.*wrong.*crop',
                r'leaf.*problem',
                r'plant.*sick',
                r'पौधे.*बीमारी',  # Hindi
                r'फसल.*रोग'
            ],
            'irrigation': [
                r'when.*water',
                r'irrigation.*schedule',
                r'how.*much.*water',
                r'watering.*time',
                r'पानी.*कब',  # Hindi
                r'सिंचाई.*समय'
            ],
            'market_prices': [
                r'price.*crop',
                r'market.*rate',
                r'sell.*crop',
                r'cost.*quintal',
                r'भाव.*दर',  # Hindi
                r'मंडी.*रेट'
            ],
            'weather': [
                r'weather.*forecast',
                r'rain.*prediction',
                r'temperature.*today',
                r'मौसम.*जानकारी',  # Hindi
                r'बारिश.*होगी'
            ],
            'soil_analysis': [
                r'soil.*test',
                r'soil.*health',
                r'fertilizer.*need',
                r'मिट्टी.*जांच',  # Hindi
                r'खाद.*जरूरत'
            ]
        }
        
        # Common agricultural entities
        self.entity_patterns = {
            'crops': [
                'wheat', 'rice', 'corn', 'tomato', 'potato', 'onion', 'cotton',
                'गेहूं', 'चावल', 'मक्का', 'टमाटर', 'आलू', 'प्याज'  # Hindi
            ],
            'seasons': [
                'kharif', 'rabi', 'zaid', 'summer', 'winter', 'monsoon',
                'खरीफ', 'रबी', 'जायद', 'गर्मी', 'सर्दी', 'बारिश'  # Hindi
            ],
            'locations': [
                'delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'punjab',
                'दिल्ली', 'मुंबई', 'कोलकाता'  # Hindi
            ]
        }
    
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition"""
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"Loaded Whisper model: {self.whisper_model_name}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
    async def process_voice_query(self, audio_file_path: str, farm_id: Optional[str] = None,
                                session: Optional[Session] = None) -> VoiceQueryResult:
        """
        Process voice query from audio file
        
        Args:
            audio_file_path: Path to audio file
            farm_id: Optional farm ID for context
            session: Database session
            
        Returns:
            VoiceQueryResult with transcription and response
        """
        start_time = datetime.now()
        
        try:
            # Validate audio file
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Check audio duration
            duration = self._get_audio_duration(audio_file_path)
            if duration > self.max_audio_duration:
                raise ValueError(f"Audio too long: {duration}s (max: {self.max_audio_duration}s)")
            
            # Transcribe audio
            transcription_result = await self._transcribe_audio(audio_file_path)
            
            # Analyze intent
            intent_result = self._analyze_intent(transcription_result['text'])
            
            # Generate response based on intent
            agent_response = await self._generate_response(
                intent_result, transcription_result, farm_id, session
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = VoiceQueryResult(
                transcribed_text=transcription_result['text'],
                detected_language=transcription_result['language'],
                confidence_score=transcription_result.get('confidence', 0.0),
                intent=intent_result.intent_type,
                entities=intent_result.entities,
                agent_response=agent_response,
                processing_time_seconds=processing_time
            )
            
            # Save to database
            if session:
                await self._save_voice_query(audio_file_path, result, farm_id, session)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VoiceQueryResult(
                transcribed_text="",
                detected_language="unknown",
                confidence_score=0.0,
                intent="error",
                entities={},
                agent_response=f"Sorry, I couldn't process your query: {str(e)}",
                processing_time_seconds=processing_time
            )
    
    def _get_audio_duration(self, audio_file_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 30.0  # Default assumption
    
    async def _transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                raise ValueError("Whisper model not loaded")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=None,  # Auto-detect language
                task='transcribe'
            )
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'confidence': self._calculate_transcription_confidence(result)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                'text': "",
                'language': 'unknown',
                'confidence': 0.0
            }
    
    def _calculate_transcription_confidence(self, whisper_result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            # Whisper doesn't directly provide confidence scores
            # We estimate based on segment information
            segments = whisper_result.get('segments', [])
            
            if not segments:
                return 0.5  # Default confidence
            
            # Calculate average confidence from segments (if available)
            confidences = []
            for segment in segments:
                # Some Whisper versions include confidence in segments
                if 'confidence' in segment:
                    confidences.append(segment['confidence'])
                else:
                    # Estimate based on segment characteristics
                    text_length = len(segment.get('text', ''))
                    duration = segment.get('end', 0) - segment.get('start', 0)
                    
                    # Longer, slower speech typically more confident
                    if duration > 0:
                        speech_rate = text_length / duration
                        confidence = min(1.0, max(0.3, 1.0 - abs(speech_rate - 10) / 20))
                        confidences.append(confidence)
            
            return sum(confidences) / len(confidences) if confidences else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _analyze_intent(self, text: str) -> QueryIntent:
        """Analyze user intent from transcribed text"""
        try:
            text_lower = text.lower()
            best_intent = 'general'
            best_confidence = 0.0
            
            # Match against intent patterns
            for intent_type, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        # Simple confidence based on pattern match
                        confidence = 0.8 if len(pattern) > 10 else 0.6
                        if confidence > best_confidence:
                            best_intent = intent_type
                            best_confidence = confidence
            
            # Extract entities
            entities = self._extract_entities(text_lower)
            
            return QueryIntent(
                intent_type=best_intent,
                confidence=best_confidence,
                entities=entities
            )
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return QueryIntent(
                intent_type='general',
                confidence=0.0,
                entities={}
            )
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities like crop names, locations from text"""
        entities = {}
        
        try:
            # Extract crops
            crops_found = []
            for crop in self.entity_patterns['crops']:
                if crop.lower() in text:
                    crops_found.append(crop)
            if crops_found:
                entities['crops'] = crops_found
            
            # Extract seasons
            seasons_found = []
            for season in self.entity_patterns['seasons']:
                if season.lower() in text:
                    seasons_found.append(season)
            if seasons_found:
                entities['seasons'] = seasons_found
            
            # Extract locations
            locations_found = []
            for location in self.entity_patterns['locations']:
                if location.lower() in text:
                    locations_found.append(location)
            if locations_found:
                entities['locations'] = locations_found
            
            # Extract numbers (for quantities, prices, etc.)
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                entities['numbers'] = [float(n) for n in numbers]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    async def _generate_response(self, intent: QueryIntent, transcription: Dict[str, Any],
                               farm_id: Optional[str], session: Optional[Session]) -> str:
        """Generate response based on detected intent"""
        try:
            intent_type = intent.intent_type
            entities = intent.entities
            language = transcription.get('language', 'en')
            
            # Route to appropriate agent based on intent
            if intent_type == 'crop_recommendation':
                return await self._handle_crop_recommendation(entities, farm_id, session, language)
            
            elif intent_type == 'disease_detection':
                return await self._handle_disease_query(entities, language)
            
            elif intent_type == 'irrigation':
                return await self._handle_irrigation_query(entities, farm_id, session, language)
            
            elif intent_type == 'market_prices':
                return await self._handle_market_query(entities, language)
            
            elif intent_type == 'weather':
                return await self._handle_weather_query(entities, farm_id, session, language)
            
            elif intent_type == 'soil_analysis':
                return await self._handle_soil_query(entities, farm_id, session, language)
            
            else:
                return self._generate_general_response(transcription['text'], language)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't understand your query. Please try again."
    
    async def _handle_crop_recommendation(self, entities: Dict[str, Any], farm_id: Optional[str],
                                        session: Optional[Session], language: str) -> str:
        """Handle crop recommendation queries"""
        try:
            if language == 'hi':
                base_response = "फसल की सिफारिश के लिए: "
            else:
                base_response = "For crop recommendations: "
            
            if not farm_id:
                if language == 'hi':
                    return base_response + "कृपया अपनी खेत की जानकारी प्रदान करें।"
                else:
                    return base_response + "Please provide your farm information first."
            
            # Get basic recommendation
            crops = entities.get('crops', [])
            seasons = entities.get('seasons', [])
            
            if crops:
                crop_name = crops[0]
                if language == 'hi':
                    return f"{crop_name} के लिए: मिट्टी की जांच कराएं, मौसम की जानकारी लें, और बाजार की दरें देखें।"
                else:
                    return f"For {crop_name}: Check soil conditions, weather forecast, and market prices before planting."
            
            if language == 'hi':
                return "मौसम और मिट्टी के आधार पर गेहूं, चावल, या सब्जियों की खेती करें।"
            else:
                return "Based on season and soil, consider wheat, rice, or vegetables. Check with agricultural extension officer for specific recommendations."
                
        except Exception as e:
            logger.error(f"Error handling crop recommendation: {e}")
            return "Please consult with agricultural experts for crop recommendations."
    
    async def _handle_disease_query(self, entities: Dict[str, Any], language: str) -> str:
        """Handle disease detection queries"""
        if language == 'hi':
            return "पौधों की बीमारी की जांच के लिए पत्तियों की तस्वीर भेजें। सामान्य उपचार: नीम का तेल, कॉपर सल्फेट का छिड़काव।"
        else:
            return "For disease detection, please upload clear photos of affected leaves. General treatment: neem oil spray, copper sulfate application."
    
    async def _handle_irrigation_query(self, entities: Dict[str, Any], farm_id: Optional[str],
                                     session: Optional[Session], language: str) -> str:
        """Handle irrigation queries"""
        if language == 'hi':
            return "सिंचाई की सलाह: सुबह या शाम को पानी दें। मिट्टी की नमी जांचें। बारिश से पहले पानी न दें।"
        else:
            return "Irrigation advice: Water early morning or evening. Check soil moisture first. Avoid watering before expected rain."
    
    async def _handle_market_query(self, entities: Dict[str, Any], language: str) -> str:
        """Handle market price queries"""
        crops = entities.get('crops', ['general crops'])
        crop_name = crops[0] if crops else 'crops'
        
        if language == 'hi':
            return f"{crop_name} की दरें मंडी में देखें। ऑनलाइन एग्रीमार्केट की जांच करें। स्थानीय व्यापारियों से संपर्क करें।"
        else:
            return f"Check mandi prices for {crop_name}. Visit online agri-market platforms. Contact local traders for current rates."
    
    async def _handle_weather_query(self, entities: Dict[str, Any], farm_id: Optional[str],
                                  session: Optional[Session], language: str) -> str:
        """Handle weather queries"""
        if language == 'hi':
            return "मौसम की जानकारी के लिए मौसम विभाग की वेबसाइट देखें। बारिश की संभावना हो तो फसल की सुरक्षा करें।"
        else:
            return "Check weather department forecasts. Protect crops if rain or storms are expected. Plan farm activities accordingly."
    
    async def _handle_soil_query(self, entities: Dict[str, Any], farm_id: Optional[str],
                               session: Optional[Session], language: str) -> str:
        """Handle soil analysis queries"""
        if language == 'hi':
            return "मिट्टी की जांच कृषि विभाग से कराएं। pH, नाइट्रोजन, फास्फोरस की जांच जरूरी है। जैविक खाद का उपयोग करें।"
        else:
            return "Get soil tested from agriculture department. Check pH, nitrogen, phosphorus levels. Use organic fertilizers for soil health."
    
    def _generate_general_response(self, text: str, language: str) -> str:
        """Generate general response for unrecognized queries"""
        if language == 'hi':
            return "मैं कृषि सहायक हूं। फसल, सिंचाई, बीमारी, मंडी की दरों के बारे में पूछ सकते हैं।"
        else:
            return "I'm an agricultural assistant. You can ask about crops, irrigation, diseases, and market prices."
    
    async def _save_voice_query(self, audio_file_path: str, result: VoiceQueryResult,
                              farm_id: Optional[str], session: Session):
        """Save voice query to database"""
        try:
            voice_query = VoiceQuery(
                farm_id=farm_id,
                audio_file_path=audio_file_path,
                transcribed_text=result.transcribed_text,
                detected_language=result.detected_language,
                confidence_score=result.confidence_score,
                agent_response=result.agent_response,
                processing_time_seconds=result.processing_time_seconds
            )
            
            session.add(voice_query)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving voice query: {e}")
            session.rollback()
    
    def process_text_query(self, text: str, farm_id: Optional[str] = None) -> str:
        """Process text query (for testing without audio)"""
        try:
            # Analyze intent
            intent_result = self._analyze_intent(text)
            
            # Generate response
            transcription = {'text': text, 'language': 'en'}
            response = self._generate_response(intent_result, transcription, farm_id, None)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text query: {e}")
            return "Sorry, I couldn't process your query."
    
    def get_supported_commands(self, language: str = 'en') -> List[str]:
        """Get list of supported voice commands"""
        if language == 'hi':
            return [
                "कौन सी फसल उगाएं?",
                "पौधे में बीमारी है",
                "कब पानी दें?",
                "मंडी में दर क्या है?",
                "मौसम कैसा रहेगा?",
                "मिट्टी की जांच कैसे करें?"
            ]
        else:
            return [
                "What crop should I grow?",
                "My plant has a disease",
                "When should I water?",
                "What are the market prices?",
                "What's the weather forecast?",
                "How to test soil health?"
            ]