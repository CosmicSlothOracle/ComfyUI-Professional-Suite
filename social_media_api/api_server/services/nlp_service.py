"""
NLP Analysis Service
Handles natural language processing for social media trend analysis
"""

import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio

# NLP libraries
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk_available = True
except ImportError:
    nltk_available = False

try:
    import spacy
    spacy_available = True
except ImportError:
    spacy_available = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    sklearn_available = True
except ImportError:
    sklearn_available = False


class NLPService:
    """Service for NLP analysis of social media trends"""

    def __init__(self):
        """Initialize the NLP service"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = "social_media_api/temp/nlp_cache"
        self.analysis_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize NLP components if available
        self.nlp_initialized = False
        self.sentiment_analyzer = None
        self.nlp_model = None

        # Try to initialize NLP components
        self._initialize_nlp()

        # Load cached analyses
        self._load_cached_analyses()

    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Initialize NLTK components
            if nltk_available:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('vader_lexicon', quiet=True)
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    self.logger.info("NLTK components initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize NLTK: {e}")

            # Initialize spaCy model
            if spacy_available:
                try:
                    # Use a smaller model for efficiency
                    self.nlp_model = spacy.load("en_core_web_sm")
                    self.logger.info("spaCy model loaded")
                except Exception as e:
                    self.logger.error(f"Failed to load spaCy model: {e}")

            self.nlp_initialized = (
                self.sentiment_analyzer is not None or self.nlp_model is not None)

        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")
            self.nlp_initialized = False

    def _load_cached_analyses(self):
        """Load cached NLP analyses from disk"""
        try:
            cache_files = os.listdir(self.cache_dir)
            for file in cache_files:
                if file.endswith(".json"):
                    with open(os.path.join(self.cache_dir, file), 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                        if "trend_id" in analysis_data:
                            self.analysis_cache[analysis_data["trend_id"]
                                                ] = analysis_data

            self.logger.info(
                f"Loaded {len(self.analysis_cache)} cached NLP analyses")
        except Exception as e:
            self.logger.error(f"Error loading cached NLP analyses: {e}")

    async def analyze_trend(self, trend_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze trend data using NLP techniques

        Args:
            trend_data: Dictionary containing trend information

        Returns:
            Dictionary with NLP analysis results or None if analysis fails
        """
        if not trend_data or "id" not in trend_data:
            self.logger.error("Invalid trend data provided for NLP analysis")
            return None

        trend_id = trend_data["id"]

        # Check if we have a cached analysis for this trend
        if trend_id in self.analysis_cache:
            self.logger.info(f"Using cached NLP analysis for trend {trend_id}")
            return self.analysis_cache[trend_id]

        self.logger.info(
            f"Performing NLP analysis for trend: {trend_data.get('name', 'Unknown')}")

        try:
            # If NLP components are not available, return basic analysis
            if not self.nlp_initialized:
                self.logger.warning(
                    "NLP components not initialized, performing basic analysis only")
                return await self._perform_basic_analysis(trend_data)

            # Prepare text for analysis
            text_content = self._extract_text_content(trend_data)

            if not text_content:
                self.logger.warning("No text content found for NLP analysis")
                return await self._perform_basic_analysis(trend_data)

            # Perform sentiment analysis
            sentiment = await self._analyze_sentiment(text_content)

            # Perform content classification
            classification = await self._classify_content(text_content, trend_data)

            # Identify content mechanics
            mechanics = await self._identify_content_mechanics(trend_data)

            # Create analysis result
            analysis_result = {
                "trend_id": trend_id,
                "analysis_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "sentiment": sentiment,
                "classification": classification,
                "content_mechanics": mechanics,
                "keywords": await self._extract_keywords(text_content),
                "summary": await self._generate_summary(trend_data, sentiment, classification)
            }

            # Cache the analysis
            self.analysis_cache[trend_id] = analysis_result

            # Save to disk
            cache_path = os.path.join(
                self.cache_dir, f"{trend_id}_analysis.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error performing NLP analysis: {e}")
            return None

    def _extract_text_content(self, trend_data: Dict[str, Any]) -> str:
        """Extract text content from trend data for analysis"""
        text_parts = []

        # Add name and description
        if "name" in trend_data:
            text_parts.append(trend_data["name"])

        if "description" in trend_data:
            text_parts.append(trend_data["description"])

        # Add example captions if available
        if "examples" in trend_data:
            for example in trend_data["examples"]:
                if "caption" in example:
                    text_parts.append(example["caption"])

        # Add hashtags if available
        if "hashtags" in trend_data:
            text_parts.extend(trend_data["hashtags"])

        return " ".join(text_parts)

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(text)

                # Determine overall sentiment
                compound = scores["compound"]
                if compound >= 0.05:
                    overall = "positive"
                elif compound <= -0.05:
                    overall = "negative"
                else:
                    overall = "neutral"

                return {
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                    "compound": compound,
                    "overall": overall
                }
            else:
                # Basic fallback if NLTK not available
                return {
                    "overall": "neutral",
                    "note": "Sentiment analysis not available"
                }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {"overall": "unknown", "error": str(e)}

    async def _classify_content(self, text: str, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify content type and theme

        Args:
            text: Text to analyze
            trend_data: Original trend data

        Returns:
            Dictionary with content classification
        """
        try:
            # Define content types and their keywords
            content_types = {
                "educational": ["learn", "education", "how to", "tutorial", "tips", "advice", "guide"],
                "entertainment": ["fun", "funny", "laugh", "comedy", "entertainment", "amusing"],
                "inspirational": ["inspire", "motivation", "success", "achieve", "dream", "goals"],
                "informative": ["news", "update", "information", "facts", "report", "data"],
                "promotional": ["promotion", "sale", "discount", "offer", "buy", "product"],
                "lifestyle": ["life", "lifestyle", "daily", "routine", "living", "experience"],
                "challenge": ["challenge", "try", "attempt", "dare", "test"],
                "storytelling": ["story", "experience", "journey", "narrative", "tale"],
                "reaction": ["reaction", "respond", "reacting", "react to"],
                "review": ["review", "opinion", "thoughts", "rating", "recommend"]
            }

            # Count matches for each content type
            scores = {}
            text_lower = text.lower()

            for content_type, keywords in content_types.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[content_type] = score

            # Get the top content types
            top_types = sorted(
                scores.items(), key=lambda x: x[1], reverse=True)
            primary_type = top_types[0][0] if top_types and top_types[0][1] > 0 else "unclassified"

            # Determine if it's ironic/satirical
            ironic_keywords = ["irony", "ironic", "sarcasm",
                               "sarcastic", "parody", "satirical", "satire"]
            ironic_score = sum(
                1 for keyword in ironic_keywords if keyword in text_lower)
            is_ironic = ironic_score > 0

            # Check platform-specific patterns
            platform = trend_data.get("platform", "").lower()

            if platform == "tiktok":
                # TikTok-specific patterns
                if "dance" in text_lower or "choreography" in text_lower:
                    primary_type = "dance"
                elif "duet" in text_lower:
                    primary_type = "duet"
                elif "pov" in text_lower:
                    primary_type = "pov"

            return {
                "primary_type": primary_type,
                "secondary_types": [t[0] for t in top_types[1:3] if t[1] > 0],
                "is_ironic": is_ironic,
                "confidence": 0.7  # Placeholder - would be calculated based on actual model confidence
            }

        except Exception as e:
            self.logger.error(f"Error in content classification: {e}")
            return {"primary_type": "unknown", "error": str(e)}

    async def _identify_content_mechanics(self, trend_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify content mechanics used in the trend

        Args:
            trend_data: Trend data

        Returns:
            List of identified content mechanics
        """
        mechanics = []
        platform = trend_data.get("platform", "").lower()
        trend_type = trend_data.get("type", "").lower()
        description = trend_data.get("description", "").lower()

        # Check for common mechanics
        if "transition" in description:
            mechanics.append({
                "type": "transition",
                "description": "Video uses smooth transitions between scenes",
                "confidence": 0.8
            })

        if "jumpcut" in description or "jump cut" in description:
            mechanics.append({
                "type": "jumpcut",
                "description": "Video uses jump cuts for pacing",
                "confidence": 0.9
            })

        if "caption" in description or "text overlay" in description:
            mechanics.append({
                "type": "captioning",
                "description": "Video uses text overlays/captions",
                "confidence": 0.85
            })

        if "voiceover" in description:
            mechanics.append({
                "type": "voiceover",
                "description": "Video uses voiceover narration",
                "confidence": 0.9
            })

        # Platform-specific mechanics
        if platform == "tiktok":
            if "duet" in description:
                mechanics.append({
                    "type": "duet",
                    "description": "TikTok duet format",
                    "confidence": 0.95
                })

            if "stitch" in description:
                mechanics.append({
                    "type": "stitch",
                    "description": "TikTok stitch format",
                    "confidence": 0.95
                })

        # If no mechanics detected, add a generic one based on trend type
        if not mechanics:
            if trend_type == "hashtag":
                mechanics.append({
                    "type": "hashtag_challenge",
                    "description": "Generic hashtag challenge format",
                    "confidence": 0.6
                })
            elif trend_type == "sound":
                mechanics.append({
                    "type": "sound_sync",
                    "description": "Content synchronized to trending sound",
                    "confidence": 0.6
                })

        return mechanics

    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        try:
            if not text:
                return []

            if self.nlp_model:
                # Use spaCy for keyword extraction
                doc = self.nlp_model(text)

                # Extract nouns and proper nouns as keywords
                keywords = [token.text.lower() for token in doc if (
                    token.pos_ in ["NOUN", "PROPN"]) and len(token.text) > 2]

                # Remove duplicates and limit to top 10
                unique_keywords = list(dict.fromkeys(keywords))
                return unique_keywords[:10]

            elif nltk_available:
                # Fallback to NLTK if spaCy not available
                tokens = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))

                # Filter out stop words and short words
                keywords = [
                    word for word in tokens if word not in stop_words and len(word) > 2]

                # Remove duplicates and limit to top 10
                unique_keywords = list(dict.fromkeys(keywords))
                return unique_keywords[:10]

            else:
                # Very basic fallback if neither is available
                words = text.lower().split()
                keywords = [word for word in words if len(word) > 3]
                unique_keywords = list(dict.fromkeys(keywords))
                return unique_keywords[:10]

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []

    async def _generate_summary(
        self,
        trend_data: Dict[str, Any],
        sentiment: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> str:
        """Generate a summary of the trend analysis"""
        try:
            trend_name = trend_data.get("name", "This trend")
            platform = trend_data.get("platform", "social media")
            trend_type = trend_data.get("type", "content")

            sentiment_text = sentiment.get("overall", "neutral")
            content_type = classification.get("primary_type", "content")

            summary = f"{trend_name} is a {sentiment_text} {content_type} {trend_type} trend on {platform}."

            if "target_demographics" in trend_data:
                demo = trend_data["target_demographics"]
                age_min = demo.get("age_min", 0)
                age_max = demo.get("age_max", 0)

                if age_min and age_max:
                    summary += f" It primarily targets users aged {age_min}-{age_max}."

            if classification.get("is_ironic", False):
                summary += " The content appears to be ironic or satirical in nature."

            return summary

        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return "Analysis summary unavailable."

    async def _perform_basic_analysis(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic analysis when NLP components are not available

        Args:
            trend_data: Trend data

        Returns:
            Basic analysis dictionary
        """
        trend_id = trend_data.get("id", str(uuid.uuid4()))

        # Create a basic analysis result
        basic_analysis = {
            "trend_id": trend_id,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "overall": "neutral",
                "note": "Sentiment analysis not available"
            },
            "classification": {
                "primary_type": trend_data.get("type", "unclassified"),
                "secondary_types": [],
                "is_ironic": False,
                "confidence": 0.5
            },
            "content_mechanics": [
                {
                    "type": trend_data.get("type", "general"),
                    "description": f"Generic {trend_data.get('type', 'content')} format",
                    "confidence": 0.5
                }
            ],
            "keywords": [trend_data.get("name", "").replace("#", "")],
            "summary": f"Basic analysis of {trend_data.get('name', 'trend')} on {trend_data.get('platform', 'social media')}."
        }

        # Cache the analysis
        self.analysis_cache[trend_id] = basic_analysis

        return basic_analysis

    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID"""
        for analysis in self.analysis_cache.values():
            if analysis.get("analysis_id") == analysis_id:
                return analysis

        return None
