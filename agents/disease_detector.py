"""
Disease Detector Agent
Uses Vision Transformers to detect crop diseases from images
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from transformers import ViTImageProcessor, ViTForImageClassification
from sqlalchemy.orm import Session

from db.models import DiseaseDetection, Farm

logger = logging.getLogger(__name__)

@dataclass
class DiseaseDetectionResult:
    disease_name: str
    confidence_score: float
    severity_level: str  # low, medium, high
    affected_area_percentage: Optional[float]
    treatment_recommendations: List[str]
    prevention_measures: List[str]
    economic_impact: Optional[str]

class DiseaseDetectorAgent:
    """
    AI-powered crop disease detection using Vision Transformers
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get('model_path', 'models/disease_model.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.75)
        self.supported_crops = config.get('supported_crops', [
            'tomato', 'potato', 'corn', 'wheat', 'rice', 'cotton'
        ])
        self.image_size = config.get('image_size', [224, 224])
        
        # Disease knowledge base
        self.disease_knowledge = self._load_disease_knowledge()
        
        # Initialize model
        self.model = None
        self.processor = None
        self._load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the disease detection model"""
        try:
            if os.path.exists(self.model_path):
                # Load custom trained model
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                logger.info(f"Loaded custom disease detection model from {self.model_path}")
            else:
                # Use pre-trained ViT model as fallback
                model_name = "google/vit-base-patch16-224"
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                self.model = ViTForImageClassification.from_pretrained(model_name)
                
                # Adapt for disease classification (this would need proper training)
                num_classes = len(self._get_disease_classes())
                self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)
                
                logger.info("Using pre-trained ViT model (requires fine-tuning for disease detection)")
                
        except Exception as e:
            logger.error(f"Error loading disease detection model: {e}")
            self.model = None
    
    def _get_disease_classes(self) -> List[str]:
        """Get list of disease classes the model can detect"""
        return [
            'healthy',
            'bacterial_spot',
            'early_blight',
            'late_blight',
            'leaf_mold',
            'septoria_leaf_spot',
            'spider_mites',
            'target_spot',
            'yellow_leaf_curl_virus',
            'mosaic_virus',
            'bacterial_canker',
            'powdery_mildew',
            'downy_mildew',
            'rust',
            'anthracnose',
            'black_rot',
            'common_scab',
            'ring_rot'
        ]
    
    def _load_disease_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load disease knowledge base with treatment recommendations"""
        return {
            'bacterial_spot': {
                'description': 'Bacterial infection causing dark spots on leaves and fruit',
                'symptoms': ['Dark spots with yellow halos', 'Leaf yellowing', 'Fruit lesions'],
                'treatment': [
                    'Apply copper-based bactericides',
                    'Remove infected plant debris',
                    'Improve air circulation',
                    'Avoid overhead watering'
                ],
                'prevention': [
                    'Use disease-resistant varieties',
                    'Crop rotation',
                    'Proper plant spacing',
                    'Drip irrigation instead of sprinklers'
                ],
                'severity_indicators': {
                    'low': 'Few scattered spots, <10% leaf area affected',
                    'medium': 'Moderate spotting, 10-30% leaf area affected',
                    'high': 'Extensive spotting, >30% leaf area affected, fruit damage'
                }
            },
            'early_blight': {
                'description': 'Fungal disease causing brown spots with concentric rings',
                'symptoms': ['Brown spots with target-like rings', 'Lower leaf yellowing', 'Stem lesions'],
                'treatment': [
                    'Apply fungicides (chlorothalonil, mancozeb)',
                    'Remove affected leaves',
                    'Improve plant nutrition',
                    'Ensure proper drainage'
                ],
                'prevention': [
                    'Crop rotation (3-4 years)',
                    'Mulching to prevent soil splash',
                    'Adequate plant spacing',
                    'Balanced fertilization'
                ],
                'severity_indicators': {
                    'low': 'Few spots on lower leaves only',
                    'medium': 'Spots spreading to middle leaves',
                    'high': 'Extensive defoliation, stem and fruit infection'
                }
            },
            'late_blight': {
                'description': 'Devastating fungal disease that can destroy entire crops',
                'symptoms': ['Water-soaked lesions', 'White fuzzy growth on leaf undersides', 'Rapid plant death'],
                'treatment': [
                    'Apply systemic fungicides immediately',
                    'Remove and destroy infected plants',
                    'Improve air circulation',
                    'Emergency harvest if necessary'
                ],
                'prevention': [
                    'Use resistant varieties',
                    'Avoid overhead irrigation',
                    'Monitor weather conditions',
                    'Preventive fungicide applications'
                ],
                'severity_indicators': {
                    'low': 'Few lesions, contained to small area',
                    'medium': 'Spreading lesions, multiple plants affected',
                    'high': 'Rapid spread, plant death, crop loss imminent'
                }
            },
            'powdery_mildew': {
                'description': 'Fungal disease creating white powdery coating on leaves',
                'symptoms': ['White powdery patches', 'Leaf curling', 'Stunted growth'],
                'treatment': [
                    'Apply sulfur-based fungicides',
                    'Neem oil applications',
                    'Baking soda spray (home remedy)',
                    'Improve air circulation'
                ],
                'prevention': [
                    'Avoid overhead watering',
                    'Proper plant spacing',
                    'Remove infected debris',
                    'Choose resistant varieties'
                ],
                'severity_indicators': {
                    'low': 'Small patches on few leaves',
                    'medium': 'Moderate coverage, multiple leaves affected',
                    'high': 'Extensive coverage, severe leaf distortion'
                }
            }
        }
    
    async def detect_disease(self, image_path: str, crop_type: str, 
                           farm_id: Optional[str] = None, 
                           session: Optional[Session] = None) -> DiseaseDetectionResult:
        """
        Detect diseases in crop images
        
        Args:
            image_path: Path to the crop image
            crop_type: Type of crop in the image
            farm_id: Optional farm ID for database storage
            session: Optional database session
            
        Returns:
            DiseaseDetectionResult with detection details
        """
        try:
            if not self.model:
                raise ValueError("Disease detection model not loaded")
            
            if crop_type not in self.supported_crops:
                logger.warning(f"Crop type '{crop_type}' not in supported crops")
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            
            # Run inference
            predictions = await self._run_inference(image)
            
            # Process predictions
            disease_name, confidence = self._process_predictions(predictions)
            
            # Determine severity
            severity = self._assess_severity(image, disease_name)
            
            # Get treatment recommendations
            treatment_info = self._get_treatment_recommendations(disease_name)
            
            # Calculate affected area (simplified approach)
            affected_area = self._estimate_affected_area(image, disease_name)
            
            # Create result
            result = DiseaseDetectionResult(
                disease_name=disease_name,
                confidence_score=confidence,
                severity_level=severity,
                affected_area_percentage=affected_area,
                treatment_recommendations=treatment_info.get('treatment', []),
                prevention_measures=treatment_info.get('prevention', []),
                economic_impact=self._assess_economic_impact(disease_name, severity)
            )
            
            # Save to database if session provided
            if session and farm_id:
                await self._save_detection_result(
                    farm_id, image_path, crop_type, result, session
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in disease detection: {e}")
            return DiseaseDetectionResult(
                disease_name="detection_error",
                confidence_score=0.0,
                severity_level="unknown",
                affected_area_percentage=None,
                treatment_recommendations=[f"Error in detection: {str(e)}"],
                prevention_measures=[],
                economic_impact="unknown"
            )
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            if self.processor:
                # Use HuggingFace processor
                inputs = self.processor(images=image, return_tensors="pt")
                return inputs['pixel_values']
            else:
                # Use custom transforms
                return self.transform(image).unsqueeze(0)
                
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    async def _run_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference on preprocessed image"""
        try:
            with torch.no_grad():
                if hasattr(self.model, 'logits'):
                    # HuggingFace model
                    outputs = self.model(image_tensor)
                    return outputs.logits
                else:
                    # Custom model
                    return self.model(image_tensor)
                    
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise
    
    def _process_predictions(self, predictions: torch.Tensor) -> Tuple[str, float]:
        """Process model predictions to get disease name and confidence"""
        try:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(predictions, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            disease_classes = self._get_disease_classes()
            disease_name = disease_classes[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # If confidence is below threshold, classify as uncertain
            if confidence_score < self.confidence_threshold:
                disease_name = "uncertain_diagnosis"
            
            return disease_name, confidence_score
            
        except Exception as e:
            logger.error(f"Error processing predictions: {e}")
            return "processing_error", 0.0
    
    def _assess_severity(self, image_tensor: torch.Tensor, disease_name: str) -> str:
        """Assess disease severity based on image analysis"""
        try:
            if disease_name in ['healthy', 'uncertain_diagnosis', 'processing_error']:
                return 'none'
            
            # Simplified severity assessment
            # In a real implementation, this would use more sophisticated image analysis
            
            # Convert tensor back to numpy for analysis
            image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
            
            # Calculate some basic metrics (this is a simplified approach)
            # Real implementation would use computer vision techniques
            
            # For now, use a simple heuristic based on image statistics
            mean_intensity = np.mean(image_np)
            std_intensity = np.std(image_np)
            
            # Simple severity classification (would need proper training)
            if std_intensity > 0.3:
                return 'high'
            elif std_intensity > 0.2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing severity: {e}")
            return 'unknown'
    
    def _estimate_affected_area(self, image_tensor: torch.Tensor, disease_name: str) -> Optional[float]:
        """Estimate percentage of affected area in the image"""
        try:
            if disease_name in ['healthy', 'uncertain_diagnosis']:
                return 0.0
            
            # Simplified affected area estimation
            # Real implementation would use segmentation models
            
            # For now, return a rough estimate based on disease type
            severity_mapping = {
                'low': 15.0,
                'medium': 35.0,
                'high': 65.0
            }
            
            severity = self._assess_severity(image_tensor, disease_name)
            return severity_mapping.get(severity, 25.0)
            
        except Exception as e:
            logger.error(f"Error estimating affected area: {e}")
            return None
    
    def _get_treatment_recommendations(self, disease_name: str) -> Dict[str, List[str]]:
        """Get treatment recommendations for detected disease"""
        return self.disease_knowledge.get(disease_name, {
            'treatment': ['Consult agricultural extension officer', 'Apply general fungicide'],
            'prevention': ['Maintain good field hygiene', 'Monitor crops regularly']
        })
    
    def _assess_economic_impact(self, disease_name: str, severity: str) -> str:
        """Assess potential economic impact of the disease"""
        if disease_name == 'healthy':
            return 'No economic impact'
        
        impact_matrix = {
            'low': {
                'bacterial_spot': 'Minor yield loss (5-10%)',
                'early_blight': 'Minimal impact with treatment',
                'powdery_mildew': 'Low impact if treated early'
            },
            'medium': {
                'bacterial_spot': 'Moderate yield loss (15-25%)',
                'early_blight': 'Significant impact without treatment',
                'late_blight': 'High risk of crop loss',
                'powdery_mildew': 'Moderate quality reduction'
            },
            'high': {
                'bacterial_spot': 'Severe yield loss (30-50%)',
                'early_blight': 'Major crop damage expected',
                'late_blight': 'Potential total crop loss',
                'powdery_mildew': 'Severe quality and yield impact'
            }
        }
        
        return impact_matrix.get(severity, {}).get(disease_name, 'Impact assessment unavailable')
    
    async def _save_detection_result(self, farm_id: str, image_path: str, crop_type: str,
                                   result: DiseaseDetectionResult, session: Session):
        """Save disease detection result to database"""
        try:
            detection_record = DiseaseDetection(
                farm_id=farm_id,
                image_path=image_path,
                crop_type=crop_type,
                detected_disease=result.disease_name,
                confidence_score=result.confidence_score,
                severity_level=result.severity_level,
                treatment_recommendations=result.treatment_recommendations
            )
            
            session.add(detection_record)
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving detection result: {e}")
            session.rollback()
    
    def batch_detect_diseases(self, image_paths: List[str], crop_type: str) -> List[DiseaseDetectionResult]:
        """Process multiple images for disease detection"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.detect_disease(image_path, crop_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append(DiseaseDetectionResult(
                    disease_name="processing_error",
                    confidence_score=0.0,
                    severity_level="unknown",
                    affected_area_percentage=None,
                    treatment_recommendations=[f"Error processing image: {str(e)}"],
                    prevention_measures=[],
                    economic_impact="unknown"
                ))
        
        return results
    
    def generate_disease_report(self, results: List[DiseaseDetectionResult]) -> str:
        """Generate a comprehensive disease detection report"""
        if not results:
            return "No disease detection results available."
        
        report = "Disease Detection Report\n"
        report += "=" * 50 + "\n\n"
        
        # Summary statistics
        total_images = len(results)
        healthy_count = sum(1 for r in results if r.disease_name == 'healthy')
        diseased_count = total_images - healthy_count
        
        report += f"Total Images Analyzed: {total_images}\n"
        report += f"Healthy Plants: {healthy_count} ({healthy_count/total_images*100:.1f}%)\n"
        report += f"Diseased Plants: {diseased_count} ({diseased_count/total_images*100:.1f}%)\n\n"
        
        # Disease breakdown
        if diseased_count > 0:
            disease_counts = {}
            for result in results:
                if result.disease_name != 'healthy':
                    disease_counts[result.disease_name] = disease_counts.get(result.disease_name, 0) + 1
            
            report += "Disease Breakdown:\n"
            for disease, count in disease_counts.items():
                report += f"- {disease.replace('_', ' ').title()}: {count} cases\n"
            
            report += "\nSeverity Analysis:\n"
            severity_counts = {'low': 0, 'medium': 0, 'high': 0}
            for result in results:
                if result.disease_name != 'healthy':
                    severity_counts[result.severity_level] = severity_counts.get(result.severity_level, 0) + 1
            
            for severity, count in severity_counts.items():
                if count > 0:
                    report += f"- {severity.title()} Severity: {count} cases\n"
        
        return report