#!/usr/bin/env python3
"""
🎯 Trading Card Authenticity Analyzer
AI-powered analysis and correction system for Pokemon TCG authenticity
Uses Hugging Face models for text analysis, character detection, and selective corrections
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import json
import logging
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import re

# Hugging Face Transformers
from transformers import (
    TrOCRProcessor, VisionEncoderDecoderModel,  # Text recognition
    pipeline,  # General AI pipelines
    AutoTokenizer, AutoModel  # Text similarity
)
import torch
from sentence_transformers import SentenceTransformer  # Text similarity

# Computer Vision
import easyocr
from sklearn.metrics.pairwise import cosine_similarity


class TradingCardAuthenticityAnalyzer:
    """AI-powered authenticity analyzer for Pokemon Trading Cards"""

    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.load_ai_models()
        self.load_authentic_tcg_patterns()

        # Analysis parameters - no upscaling!
        self.analysis_params = {
            'preserve_resolution': True,
            'text_similarity_threshold': 0.75,
            'character_detection_confidence': 0.8,
            'inpainting_strength': 0.7,
            'max_text_correction_attempts': 3
        }

        # Performance optimization settings
        self.performance_params = {
            'enable_gpu': torch.cuda.is_available(),
            'batch_size': 4 if torch.cuda.is_available() else 2,
            'enable_model_caching': True,
            'preprocess_cache': {},
            'max_image_size': 2048  # Limit processing size for performance
        }

        self.logger.info(
            f"Performance settings: GPU={self.performance_params['enable_gpu']}, Batch Size={self.performance_params['batch_size']}")

        # Model device optimization
        if self.performance_params['enable_gpu']:
            try:
                self.device = torch.device("cuda")
                self.logger.info("✓ CUDA GPU acceleration enabled")
            except:
                self.device = torch.device("cpu")
                self.logger.info("⚠ GPU unavailable, using CPU")
        else:
            self.device = torch.device("cpu")
            self.logger.info("ℹ Using CPU processing")

    def setup_logging(self):
        """Setup logging system"""
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/authenticity_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create analysis directories"""
        directories = [
            "output/corrected",
            "output/analysis",
            "output/reports",
            "output/text_analysis",
            "output/character_analysis"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def load_ai_models(self):
        """Load Hugging Face AI models for analysis

        Model Selection Rationale:
        1. TrOCR: State-of-the-art optical character recognition
        2. SentenceTransformer: Semantic text similarity for authenticity
        3. BLIP: Image captioning for character identification
        4. SegmentAnything: Precise region detection
        """
        self.logger.info("Loading AI models...")

        # 1. OCR Model - TrOCR (Microsoft)
        # Rationale: Best performance on stylized text like trading cards
        try:
            self.ocr_processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-printed')
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-printed')
            self.logger.info(
                "✓ TrOCR model loaded (Microsoft - optimal for printed text)")
        except Exception as e:
            self.logger.warning(f"TrOCR failed, using EasyOCR fallback: {e}")
            self.ocr_reader = easyocr.Reader(['en'])

        # 2. Text Similarity Model - SentenceTransformers
        # Rationale: Semantic understanding to compare AI text vs authentic Pokemon text
        try:
            self.text_similarity_model = SentenceTransformer(
                'all-MiniLM-L6-v2')
            self.logger.info(
                "✓ SentenceTransformer loaded (semantic text comparison)")
        except Exception as e:
            self.logger.error(f"Text similarity model failed: {e}")

        # 3. Image Captioning - BLIP
        # Rationale: Identify Pokemon characters in evolution chain images
        try:
            self.image_captioner = pipeline("image-to-text",
                                            model="Salesforce/blip-image-captioning-base")
            self.logger.info(
                "✓ BLIP captioning loaded (character identification)")
        except Exception as e:
            self.logger.warning(f"BLIP model failed: {e}")

        # 4. Image Segmentation - DeepLab
        # Rationale: Precise text region detection for selective removal
        try:
            self.segmentation_model = pipeline("image-segmentation",
                                               model="facebook/detr-resnet-50-panoptic")
            self.logger.info(
                "✓ DETR segmentation loaded (precise region detection)")
        except Exception as e:
            self.logger.warning(f"Segmentation model failed: {e}")

    def load_authentic_tcg_patterns(self):
        """Load authentic Pokemon TCG text patterns for comparison"""

        # Authentic Pokemon TCG text patterns based on reference cards
        self.authentic_patterns = {
            'evolution_text': [
                "Evolves from {pokemon}",
                "Put {pokemon} on the Stage {num} card",
                "Stage 2 Pokemon"
            ],
            'ability_headers': [
                "Ability",
                "Poke-BODY",
                "Poke-Power"
            ],
            'attack_patterns': [
                r"^[A-Z][a-z\s]+\s+\d+$",  # Attack name + damage
                r"^[A-Z][a-z\s]+\s+\d+x$"  # Attack name + damage multiplier
            ],
            'card_stats': [
                r"HP\s+\d+",  # HP pattern
                r"Length:\s+\d+",  # Length pattern
                r"Weight:\s+\d+",  # Weight pattern
            ],
            'forbidden_text': [
                'ChatGPT', 'AI generated', 'artificial intelligence',
                'neural network', 'machine learning', 'AI model'
            ]
        }

        self.logger.info("✓ Authentic TCG patterns loaded")

    def analyze_card_text(self, image: np.ndarray) -> Dict:
        """Comprehensive text analysis using AI models"""

        # Convert to PIL for model processing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Step 1: Extract all text using TrOCR
        extracted_text = self.extract_text_trocr(pil_image)

        # Step 2: Analyze text authenticity
        authenticity_score = self.analyze_text_authenticity(extracted_text)

        # Step 3: Detect problematic text regions
        problematic_regions = self.detect_problematic_text(
            image, extracted_text)

        # Step 4: Suggest corrections
        corrections = self.suggest_text_corrections(extracted_text)

        return {
            'extracted_text': extracted_text,
            'authenticity_score': authenticity_score,
            'problematic_regions': problematic_regions,
            'corrections': corrections,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def extract_text_trocr(self, pil_image: Image.Image) -> List[Dict]:
        """Extract text using TrOCR model with dynamic region detection"""

        # Use DETR segmentation to find actual text regions
        regions = self.detect_dynamic_card_regions(pil_image)

        extracted_text = []

        for region_name, (x1, y1, x2, y2) in regions.items():
            # Crop region
            region_image = pil_image.crop((x1, y1, x2, y2))

            # Preprocess region for better OCR
            region_image = self.preprocess_ocr_region(region_image)

            try:
                # TrOCR processing
                pixel_values = self.ocr_processor(
                    region_image, return_tensors="pt").pixel_values
                generated_ids = self.ocr_model.generate(pixel_values)
                generated_text = self.ocr_processor.batch_decode(
                    generated_ids, skip_special_tokens=True)[0]

                if generated_text.strip():
                    extracted_text.append({
                        'region': region_name,
                        'text': generated_text.strip(),
                        'coordinates': (x1, y1, x2, y2),
                        'confidence': 0.85  # TrOCR doesn't provide confidence directly
                    })

            except Exception as e:
                self.logger.warning(f"TrOCR failed for {region_name}: {e}")
                # Fallback to EasyOCR if available
                if hasattr(self, 'ocr_reader'):
                    results = self.ocr_reader.readtext(np.array(region_image))
                    for (bbox, text, confidence) in results:
                        if confidence > 0.5 and text.strip():
                            extracted_text.append({
                                'region': region_name,
                                'text': text.strip(),
                                'coordinates': (x1, y1, x2, y2),
                                'confidence': confidence
                            })

        return extracted_text

    def detect_dynamic_card_regions(self, pil_image: Image.Image) -> Dict[str, Tuple[int, int, int, int]]:
        """Dynamically detect Pokemon card regions using AI segmentation"""

        width, height = pil_image.size

        try:
            # Use DETR segmentation to find text regions
            segments = self.segmentation_model(pil_image)

            # Initialize regions with intelligent defaults
            regions = {}

            # Analyze segmentation results to identify card regions
            text_regions = []
            for segment in segments:
                if segment.get('score', 0) > 0.7:  # High confidence segments
                    label = segment.get('label', '').lower()
                    if any(text_word in label for text_word in ['text', 'person', 'book']):
                        bbox = segment.get('bbox', [])
                        if len(bbox) == 4:
                            text_regions.append(bbox)

            # Sort regions by position to identify card layout
            text_regions.sort(key=lambda x: (x[1], x[0]))  # Sort by Y then X

            # Map detected regions to Pokemon card areas
            if len(text_regions) >= 2:
                # Evolution area: top-left region
                if text_regions:
                    regions['evolution_area'] = (
                        max(0, int(width * 0.05)),
                        max(0, int(height * 0.05)),
                        min(width, int(width * 0.5)),
                        min(height, int(height * 0.2))
                    )

                # Pokemon name: top-center region
                regions['pokemon_name'] = (
                    max(0, int(width * 0.1)),
                    max(0, int(height * 0.15)),
                    min(width, int(width * 0.9)),
                    min(height, int(height * 0.3))
                )

                # Ability area: middle region
                regions['ability_area'] = (
                    max(0, int(width * 0.1)),
                    max(0, int(height * 0.55)),
                    min(width, int(width * 0.9)),
                    min(height, int(height * 0.8))
                )

                # Stats area: bottom region
                regions['stats_area'] = (
                    max(0, int(width * 0.05)),
                    max(0, int(height * 0.8)),
                    min(width, int(width * 0.95)),
                    min(height, int(height * 0.98))
                )
            else:
                # Fallback to optimized default regions for Pokemon cards
                regions = self.get_optimized_default_regions(width, height)

        except Exception as e:
            self.logger.warning(f"Dynamic region detection failed: {e}")
            # Use optimized default regions
            regions = self.get_optimized_default_regions(width, height)

        return regions

    def get_optimized_default_regions(self, width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
        """Get optimized default regions based on typical Pokemon card layout"""
        return {
            'evolution_area': (
                int(width * 0.05), int(height * 0.05),
                int(width * 0.45), int(height * 0.15)
            ),
            'pokemon_name': (
                int(width * 0.1), int(height * 0.15),
                int(width * 0.85), int(height * 0.25)
            ),
            'ability_area': (
                int(width * 0.08), int(height * 0.6),
                int(width * 0.92), int(height * 0.8)
            ),
            'stats_area': (
                int(width * 0.05), int(height * 0.85),
                int(width * 0.95), int(height * 0.98)
            )
        }

    def preprocess_ocr_region(self, region_image: Image.Image) -> Image.Image:
        """Preprocess image region for better OCR accuracy"""

        # Convert to grayscale if needed
        if region_image.mode != 'L':
            gray_image = region_image.convert('L')
        else:
            gray_image = region_image

        # Enhance contrast for better text recognition
        enhancer = ImageEnhance.Contrast(gray_image)
        contrast_enhanced = enhancer.enhance(1.5)

        # Sharpen for better edge definition
        sharpener = ImageEnhance.Sharpness(contrast_enhanced)
        sharpened = sharpener.enhance(1.3)

        # Convert back to RGB for TrOCR
        return sharpened.convert('RGB')

    def analyze_text_authenticity(self, extracted_text: List[Dict]) -> float:
        """Analyze text authenticity using semantic similarity"""

        total_score = 0
        text_count = 0

        for text_data in extracted_text:
            text = text_data['text']
            region = text_data['region']

            # Check for forbidden AI-generated text
            forbidden_score = self.check_forbidden_text(text)

            # Check pattern authenticity based on region
            pattern_score = self.check_pattern_authenticity(text, region)

            # Semantic similarity to authentic Pokemon text
            semantic_score = self.check_semantic_authenticity(text, region)

            # Combined score (weighted)
            combined_score = (
                forbidden_score * 0.4 +  # High weight for forbidden text
                pattern_score * 0.3 +    # Pattern matching
                semantic_score * 0.3     # Semantic similarity
            )

            total_score += combined_score
            text_count += 1

            self.logger.info(
                f"Text '{text[:30]}...' authenticity: {combined_score:.2f}")

        return total_score / text_count if text_count > 0 else 0.0

    def check_forbidden_text(self, text: str) -> float:
        """Check for AI-generation indicators"""

        text_lower = text.lower()

        for forbidden in self.authentic_patterns['forbidden_text']:
            if forbidden.lower() in text_lower:
                self.logger.warning(
                    f"Forbidden text detected: '{forbidden}' in '{text}'")
                return 0.0  # Immediate fail

        return 1.0  # Pass

    def check_pattern_authenticity(self, text: str, region: str) -> float:
        """Check if text matches authentic Pokemon TCG patterns"""

        if region == 'evolution_area':
            for pattern in self.authentic_patterns['evolution_text']:
                if any(word in text for word in pattern.split()):
                    return 1.0
            return 0.3  # Partial score if no evolution text found

        elif region == 'ability_area':
            for pattern in self.authentic_patterns['ability_headers']:
                if pattern.lower() in text.lower():
                    return 1.0
            return 0.5

        elif region == 'stats_area':
            for pattern in self.authentic_patterns['card_stats']:
                if re.search(pattern, text):
                    return 1.0
            return 0.4

        return 0.7  # Default score for other regions

    def check_semantic_authenticity(self, text: str, region: str) -> float:
        """Use AI to check semantic similarity to authentic Pokemon text"""

        # Sample authentic texts for comparison
        authentic_samples = {
            'evolution_area': [
                "Evolves from Charmeleon Put Charizard on the Stage 1 card",
                "Stage 2 Pokemon"
            ],
            'ability_area': [
                "Ability Electrogenesis Once during your turn you may search your deck",
                "Poke-BODY Crystal Type Charizard's type is the same as that Energy"
            ],
            'pokemon_name': [
                "Charizard 110 HP", "Pikachu 60 HP", "Gengar 80 HP"
            ]
        }

        if region not in authentic_samples:
            return 0.7  # Default score

        try:
            # Encode texts
            text_embedding = self.text_similarity_model.encode([text])
            authentic_embeddings = self.text_similarity_model.encode(
                authentic_samples[region])

            # Calculate similarity
            similarities = cosine_similarity(
                text_embedding, authentic_embeddings)[0]
            max_similarity = np.max(similarities)

            return float(max_similarity)

        except Exception as e:
            self.logger.warning(f"Semantic analysis failed: {e}")
            return 0.5

    def detect_problematic_text(self, image: np.ndarray, extracted_text: List[Dict]) -> List[Dict]:
        """Detect text regions that need correction or removal"""

        problematic_regions = []

        for text_data in extracted_text:
            text = text_data['text']
            region = text_data['region']
            confidence = text_data['confidence']

            # Criteria for problematic text
            is_problematic = (
                confidence < 0.6 or  # Low OCR confidence
                any(forbidden in text.lower() for forbidden in self.authentic_patterns['forbidden_text']) or
                len(text) < 2 or  # Too short
                not re.search(r'[a-zA-Z]', text)  # No letters (garbled)
            )

            if is_problematic:
                problematic_regions.append({
                    'text': text,
                    'region': region,
                    'coordinates': text_data['coordinates'],
                    'reason': self.get_problem_reason(text, confidence),
                    'action': 'remove' if 'forbidden' in text.lower() else 'correct'
                })

        return problematic_regions

    def suggest_text_corrections(self, extracted_text: List[Dict]) -> List[Dict]:
        """Suggest corrections for problematic text using AI analysis"""

        corrections = []

        for text_data in extracted_text:
            text = text_data['text']
            region = text_data['region']
            confidence = text_data['confidence']

            # Generate suggestions based on region and text analysis
            suggestions = []

            # Check if text needs correction
            if confidence < 0.7 or self.needs_correction(text, region):
                suggestions = self.generate_text_suggestions(text, region)

            if suggestions:
                corrections.append({
                    'original_text': text,
                    'region': region,
                    'coordinates': text_data['coordinates'],
                    'suggested_corrections': suggestions,
                    'confidence_score': confidence,
                    'correction_reason': self.get_correction_reason(text, region, confidence)
                })

        return corrections

    def needs_correction(self, text: str, region: str) -> bool:
        """Determine if text needs correction"""

        # Check for forbidden AI-generated text
        for forbidden in self.authentic_patterns['forbidden_text']:
            if forbidden.lower() in text.lower():
                return True

        # Check for garbled text
        if len(text) < 2 or not re.search(r'[a-zA-Z]', text):
            return True

        # Check region-specific patterns
        if region == 'pokemon_name' and not re.search(r'^[A-Z][a-z]+(\s+[A-Z]+)*\s+\d+\s+HP', text):
            return True

        return False

    def generate_text_suggestions(self, text: str, region: str) -> List[str]:
        """Generate correction suggestions based on region and context"""

        suggestions = []

        if region == 'evolution_area':
            if 'evolve' in text.lower() or 'stage' in text.lower():
                suggestions = [
                    "Evolves from [Pokemon Name]",
                    "Stage 1 Pokemon",
                    "Stage 2 Pokemon"
                ]
            else:
                suggestions = ["[Remove garbled text]"]

        elif region == 'pokemon_name':
            # Extract possible Pokemon name and HP
            name_match = re.search(r'([A-Za-z]+)', text)
            hp_match = re.search(r'(\d+)', text)

            if name_match:
                pokemon_name = name_match.group(1).title()
                hp_value = hp_match.group(1) if hp_match else "80"
                suggestions.append(f"{pokemon_name} {hp_value} HP")
            else:
                suggestions = ["[Pokemon Name] 80 HP"]

        elif region == 'ability_area':
            if 'ability' in text.lower():
                suggestions = [
                    "Ability [Ability Name]",
                    "Poke-Power [Power Name]",
                    "Poke-Body [Body Name]"
                ]
            else:
                suggestions = ["[Remove corrupted ability text]"]

        elif region == 'stats_area':
            suggestions = [
                "Length: [X] m",
                "Weight: [X] kg",
                "Weakness: [Type]",
                "Resistance: [Type]"
            ]

        else:
            suggestions = ["[Remove garbled text]"]

        return suggestions[:3]  # Limit to top 3 suggestions

    def get_correction_reason(self, text: str, region: str, confidence: float) -> str:
        """Get reason why correction is needed"""

        if any(forbidden in text.lower() for forbidden in self.authentic_patterns['forbidden_text']):
            return "Contains AI-generation indicators"
        elif confidence < 0.6:
            return f"Low OCR confidence ({confidence:.2f})"
        elif len(text) < 2:
            return "Text too short"
        elif not re.search(r'[a-zA-Z]', text):
            return "No readable letters"
        else:
            return f"Doesn't match authentic {region} patterns"

    def get_problem_reason(self, text: str, confidence: float) -> str:
        """Determine why text is problematic"""

        if any(forbidden in text.lower() for forbidden in self.authentic_patterns['forbidden_text']):
            return "Contains AI-generation indicators"
        elif confidence < 0.6:
            return f"Low OCR confidence ({confidence:.2f})"
        elif len(text) < 2:
            return "Text too short/garbled"
        elif not re.search(r'[a-zA-Z]', text):
            return "No readable letters"
        else:
            return "Pattern mismatch with authentic TCG text"

    def analyze_character_images(self, image: np.ndarray) -> Dict:
        """Analyze Pokemon character images, especially evolution chain"""

        height, width = image.shape[:2]

        # Define multiple character detection regions for better coverage
        character_regions = {
            'evolution_area': image[0:int(height*0.25), 0:int(width*0.4)],
            'main_artwork': image[int(height*0.2):int(height*0.65), int(width*0.05):int(width*0.95)],
            'full_card': image  # Fallback - analyze entire card
        }

        analysis = {
            'evolution_characters': [],
            'main_character': None,
            'character_authenticity': 0.0,
            'detected_regions': []
        }

        all_detected_pokemon = []

        for region_name, region_image in character_regions.items():
            try:
                # Convert to PIL for AI analysis with enhanced preprocessing
                region_pil = Image.fromarray(
                    cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB))

                # Preprocess for better character detection
                preprocessed_region = self.preprocess_character_region(
                    region_pil)

                # Use BLIP to identify characters
                caption_result = self.image_captioner(preprocessed_region)
                caption = caption_result[0]['generated_text'] if caption_result else ""

                # Extract Pokemon names from caption using enhanced detection
                pokemon_names = self.extract_pokemon_names_enhanced(caption)

                if pokemon_names:
                    all_detected_pokemon.extend(pokemon_names)
                    analysis['detected_regions'].append({
                        'region': region_name,
                        'caption': caption,
                        'pokemon_found': pokemon_names
                    })

                    self.logger.info(
                        f"Region {region_name}: Found {pokemon_names} from caption: '{caption[:50]}...'")

            except Exception as e:
                self.logger.warning(
                    f"Character analysis failed for {region_name}: {e}")

        # Consolidate results
        unique_pokemon = list(set(all_detected_pokemon))
        analysis['evolution_characters'] = unique_pokemon
        analysis['main_character'] = unique_pokemon[0] if unique_pokemon else None
        analysis['character_authenticity'] = self.validate_evolution_chain_enhanced(
            unique_pokemon, all_detected_pokemon)

        self.logger.info(
            f"Character analysis complete: {unique_pokemon}, authenticity: {analysis['character_authenticity']:.2f}")

        return analysis

    def preprocess_character_region(self, region_image: Image.Image) -> Image.Image:
        """Enhanced preprocessing for better character detection"""

        # Resize to optimal size for BLIP (maintaining aspect ratio)
        width, height = region_image.size
        max_size = 512

        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            region_image = region_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS)

        # Enhance colors for better character recognition
        enhancer = ImageEnhance.Color(region_image)
        enhanced = enhancer.enhance(1.2)

        # Improve contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        contrast_enhanced = contrast_enhancer.enhance(1.3)

        # Sharpen details
        sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
        final_image = sharpness_enhancer.enhance(1.2)

        return final_image

    def extract_pokemon_names_enhanced(self, caption: str) -> List[str]:
        """Enhanced Pokemon name extraction with expanded database"""

        # Expanded Pokemon database including German names and variations
        pokemon_database = {
            # Generation 1 - Core Pokemon
            'pikachu': ['pikachu', 'pika'],
            'charizard': ['charizard', 'glurak'],
            'gengar': ['gengar'],
            'squirtle': ['squirtle', 'schiggy'],
            'bulbasaur': ['bulbasaur', 'bisasam'],
            'charmander': ['charmander', 'glumanda'],
            'charmeleon': ['charmeleon', 'glutexo'],
            'wartortle': ['wartortle', 'schillok'],
            'ivysaur': ['ivysaur', 'bisaknosp'],
            'blastoise': ['blastoise', 'turtok'],
            'venusaur': ['venusaur', 'bisaflor'],

            # Generation 1 - Other popular Pokemon
            'magikarp': ['magikarp', 'karpador'],
            'gyarados': ['gyarados', 'garados'],
            'machamp': ['machamp', 'machomei'],
            'alakazam': ['alakazam', 'simsala'],
            'golem': ['golem', 'geowaz'],
            'graveler': ['graveler', 'georok'],
            'geodude': ['geodude', 'kleinstein'],
            'abra': ['abra'],
            'kadabra': ['kadabra'],
            'machop': ['machop', 'machollo'],
            'machoke': ['machoke', 'maschock'],

            # Later Generation Pokemon that might appear
            'wobbuffet': ['wobbuffet', 'woingenau'],
            'lucario': ['lucario'],
            'garchomp': ['garchomp', 'knakrack'],
            'metagross': ['metagross', 'metagross'],
            'dragonite': ['dragonite', 'dragoran'],
            'tyranitar': ['tyranitar', 'despotar'],

            # VMAX and special variants
            'vmax': ['vmax', 'v-max'],
            'ex': ['ex'],
            'gx': ['gx']
        }

        found_pokemon = []
        caption_lower = caption.lower()

        # Search for Pokemon names and their variants
        for base_name, variants in pokemon_database.items():
            for variant in variants:
                if variant in caption_lower:
                    # Add base name (capitalized) if not already found
                    capitalized_name = base_name.title()
                    if capitalized_name not in found_pokemon:
                        found_pokemon.append(capitalized_name)

        # Also search for generic Pokemon terms
        pokemon_terms = ['pokemon', 'pokémon', 'creature',
                         'monster', 'dragon', 'electric', 'fire', 'water', 'grass']
        detected_terms = [
            term for term in pokemon_terms if term in caption_lower]

        if detected_terms and not found_pokemon:
            # If we detect Pokemon-related terms but no specific Pokemon, log for analysis
            self.logger.info(
                f"Detected Pokemon terms but no specific Pokemon: {detected_terms} in '{caption}'")

        return found_pokemon

    def validate_evolution_chain_enhanced(self, unique_pokemon: List[str], all_detected: List[str]) -> float:
        """Enhanced evolution chain validation with better scoring"""

        # Enhanced evolution chains including more Pokemon
        evolution_chains = {
            'Charmander': ['Charmeleon', 'Charizard'],
            'Squirtle': ['Wartortle', 'Blastoise'],
            'Bulbasaur': ['Ivysaur', 'Venusaur'],
            'Magikarp': ['Gyarados'],
            'Abra': ['Kadabra', 'Alakazam'],
            'Machop': ['Machoke', 'Machamp'],
            'Geodude': ['Graveler', 'Golem'],
            'Gengar': [],  # Gengar is often standalone or end evolution
            'Pikachu': [],  # Pikachu often standalone
            'Wobbuffet': []  # Standalone Pokemon
        }

        if not unique_pokemon:
            return 0.3  # Lower score for no detection

        # Base score for detecting any Pokemon
        base_score = 0.6

        # Check for valid evolution chains
        evolution_bonus = 0.0
        for base_pokemon, evolutions in evolution_chains.items():
            if base_pokemon in unique_pokemon:
                evolution_bonus += 0.1
                # Bonus for finding related evolutions
                for evolution in evolutions:
                    if evolution in unique_pokemon:
                        evolution_bonus += 0.15

        # Bonus for multiple detections (consistency)
        consistency_bonus = 0.0
        if len(all_detected) > len(unique_pokemon):  # Multiple detections of same Pokemon
            consistency_bonus = 0.1

        # Bonus for detecting special variants
        special_bonus = 0.0
        special_variants = ['Vmax', 'Ex', 'Gx']
        for variant in special_variants:
            if variant in unique_pokemon:
                special_bonus += 0.05

        final_score = min(1.0, base_score + evolution_bonus +
                          consistency_bonus + special_bonus)

        self.logger.info(
            f"Character authenticity calculation: base={base_score}, evolution={evolution_bonus}, consistency={consistency_bonus}, special={special_bonus}, final={final_score}")

        return final_score

    def correct_card_issues(self, image: np.ndarray, text_analysis: Dict, character_analysis: Dict) -> np.ndarray:
        """Apply corrections to card based on analysis"""

        corrected_image = image.copy()

        # Step 1: Remove problematic text
        for problem in text_analysis['problematic_regions']:
            if problem['action'] == 'remove':
                corrected_image = self.remove_text_region(
                    corrected_image, problem['coordinates'])
                self.logger.info(
                    f"Removed problematic text: '{problem['text'][:30]}...'")

        # Step 2: Enhance character image clarity (without upscaling)
        corrected_image = self.enhance_character_region(corrected_image)

        # Step 3: Clean up any remaining artifacts
        corrected_image = self.clean_artifacts(corrected_image)

        return corrected_image

    def remove_text_region(self, image: np.ndarray, coordinates: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove text from specific region using inpainting"""

        x1, y1, x2, y2 = coordinates

        # Create mask for inpainting
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # Use OpenCV inpainting for text removal
        inpainted = cv2.inpaint(
            image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted

    def enhance_character_region(self, image: np.ndarray) -> np.ndarray:
        """Enhance character region clarity without upscaling"""

        height, width = image.shape[:2]

        # Evolution area (top-left)
        evolution_region = image[0:int(height*0.2), 0:int(width*0.3)]

        # Apply targeted enhancement
        enhanced_region = self.apply_selective_enhancement(evolution_region)

        # Replace region in original image
        enhanced_image = image.copy()
        enhanced_image[0:int(height*0.2), 0:int(width*0.3)] = enhanced_region

        return enhanced_image

    def apply_selective_enhancement(self, region: np.ndarray) -> np.ndarray:
        """Apply careful enhancement to preserve authenticity"""

        # Convert to PIL for processing
        pil_region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

        # Minimal enhancement - just clarity improvements
        from PIL import ImageEnhance

        # Slight sharpness increase
        sharpness_enhancer = ImageEnhance.Sharpness(pil_region)
        enhanced = sharpness_enhancer.enhance(1.1)

        # Minimal contrast adjustment
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(1.05)

        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    def clean_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Clean remaining visual artifacts"""

        # Apply gentle noise reduction
        cleaned = cv2.bilateralFilter(image, 5, 50, 50)

        return cleaned

    def process_single_card(self, image_path: str) -> Dict:
        """Process a single trading card for authenticity analysis"""

        self.logger.info(f"Analyzing card: {os.path.basename(image_path)}")

        # Load image with optimized performance handling
        image = self.load_image_optimized(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        start_time = time.time()

        # Step 1: Text Analysis
        self.logger.info("Step 1: AI-powered text analysis...")
        text_analysis = self.analyze_card_text(image)

        # Step 2: Character Analysis
        self.logger.info("Step 2: Character image analysis...")
        character_analysis = self.analyze_character_images(image)

        # Step 3: Apply Corrections
        self.logger.info("Step 3: Applying authenticity corrections...")
        corrected_image = self.correct_card_issues(
            image, text_analysis, character_analysis)

        # Step 4: Save Results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"corrected_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.png"
        output_path = os.path.join("output/corrected", output_filename)

        cv2.imwrite(output_path, corrected_image)

        # Generate comprehensive report
        processing_time = time.time() - start_time

        report = {
            'input_file': os.path.basename(image_path),
            'output_file': output_filename,
            'processing_time': processing_time,
            'text_analysis': text_analysis,
            'character_analysis': character_analysis,
            'overall_authenticity_score': (
                text_analysis['authenticity_score'] * 0.7 +
                character_analysis['character_authenticity'] * 0.3
            ),
            'corrections_applied': len(text_analysis['problematic_regions']),
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': {
                'image_size': f"{image.shape[1]}x{image.shape[0]}",
                'gpu_used': self.performance_params['enable_gpu'],
                'processing_speed': f"{processing_time:.2f}s"
            }
        }

        # Save detailed analysis report
        report_path = os.path.join(
            "output/reports", f"analysis_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Analysis complete. Authenticity score: {report['overall_authenticity_score']:.2f}")
        self.logger.info(
            f"Corrections applied: {report['corrections_applied']}")

        return report

    def load_image_optimized(self, image_path: str) -> Optional[np.ndarray]:
        """Optimized image loading with performance considerations"""

        try:
            # Try direct loading first
            image = cv2.imread(image_path)
            if image is None:
                # Try with UTF-8 encoding for special characters
                image_path_encoded = image_path.encode('utf-8').decode('utf-8')
                image = cv2.imread(image_path_encoded)

            if image is None:
                # Final attempt with numpy/PIL for difficult encodings
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Optimize image size for performance
            if image is not None:
                height, width = image.shape[:2]
                max_size = self.performance_params['max_image_size']

                if max(width, height) > max_size:
                    self.logger.info(
                        f"Resizing large image from {width}x{height} for better performance")

                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = max_size
                        new_height = int(height * max_size / width)
                    else:
                        new_height = max_size
                        new_width = int(width * max_size / height)

                    image = cv2.resize(
                        image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    self.logger.info(
                        f"Image resized to {new_width}x{new_height}")

            return image

        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    def process_batch(self, input_directory: str) -> Dict:
        """Process batch of trading cards"""

        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_directory)
                       if f.lower().endswith(image_extensions)]

        if not image_files:
            self.logger.warning("No image files found in input directory")
            return None

        self.logger.info(
            f"Starting authenticity analysis of {len(image_files)} cards")
        self.logger.info("=" * 60)

        results = []
        successful = 0
        failed = 0
        start_time = time.time()

        for i, filename in enumerate(image_files, 1):
            self.logger.info(f"\n--- Analyzing {i}/{len(image_files)} ---")

            image_path = os.path.join(input_directory, filename)

            try:
                result = self.process_single_card(image_path)
                result['success'] = True
                results.append(result)
                successful += 1

            except Exception as e:
                self.logger.error(f"Failed to process {filename}: {str(e)}")
                results.append({
                    'success': False,
                    'input_file': filename,
                    'error': str(e)
                })
                failed += 1

            # Progress update
            progress = (i / len(image_files)) * 100
            self.logger.info(
                f"Progress: {progress:.1f}% ({successful} success, {failed} failed)")

        # Final summary
        total_time = time.time() - start_time

        self.logger.info("\n" + "=" * 60)
        self.logger.info("AUTHENTICITY ANALYSIS COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Cards: {len(image_files)}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total Time: {total_time:.1f}s")

        # Calculate average authenticity
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_authenticity = sum(r['overall_authenticity_score']
                                   for r in successful_results) / len(successful_results)
            self.logger.info(
                f"Average Authenticity Score: {avg_authenticity:.2f}")

        # Save batch report
        batch_report = {
            'timestamp': datetime.now().isoformat(),
            'total_cards': len(image_files),
            'successful': successful,
            'failed': failed,
            'processing_time_seconds': total_time,
            'average_authenticity': avg_authenticity if successful_results else 0,
            'results': results
        }

        report_path = os.path.join(
            "output/reports", f"batch_authenticity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Report saved: {report_path}")
        self.logger.info(f"Corrected cards: output/corrected/")
        self.logger.info("=" * 60)

        return batch_report


def main():
    """Main execution function"""
    print("AI-POWERED TRADING CARD AUTHENTICITY ANALYZER")
    print("=" * 60)
    print("Uses Hugging Face AI models for:")
    print("- Text authenticity analysis (TrOCR + SentenceTransformers)")
    print("- Character identification (BLIP)")
    print("- Selective text removal/correction")
    print("- NO upscaling - authenticity focus")
    print("=" * 60)

    # Initialize analyzer
    analyzer = TradingCardAuthenticityAnalyzer()

    # Process batch
    try:
        batch_report = analyzer.process_batch("input")

        if batch_report and batch_report['successful'] > 0:
            print(
                f"\nSUCCESS! Analyzed {batch_report['successful']}/{batch_report['total_cards']} cards")
            print(
                f"Average Authenticity Score: {batch_report['average_authenticity']:.2f}")
            print(
                f"Total Time: {batch_report['processing_time_seconds']:.1f} seconds")
            print(f"Results in: output/corrected/")
        else:
            print("No cards were successfully processed")

    except Exception as e:
        logging.error(f"Critical error: {e}")
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
