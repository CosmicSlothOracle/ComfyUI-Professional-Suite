"""
Post Text Generation Service
Handles generation of social media post text and hashtags
"""

import logging
import json
import os
import uuid
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

# NLP libraries (optional)
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False


class PostGenerationService:
    """Service for generating social media post text"""

    def __init__(self):
        """Initialize the post generation service"""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = "social_media_api/temp/post_cache"
        self.post_cache = {}

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load cached posts
        self._load_cached_posts()

        # Load templates
        self.templates = self._load_templates()

    def _load_cached_posts(self):
        """Load cached posts from disk"""
        try:
            if os.path.exists(self.cache_dir):
                cache_files = os.listdir(self.cache_dir)
                for file in cache_files:
                    if file.endswith(".json"):
                        with open(os.path.join(self.cache_dir, file), 'r', encoding='utf-8') as f:
                            post_data = json.load(f)
                            if "post_id" in post_data:
                                self.post_cache[post_data["post_id"]
                                                ] = post_data

                self.logger.info(f"Loaded {len(self.post_cache)} cached posts")
        except Exception as e:
            self.logger.error(f"Error loading cached posts: {e}")

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load post text templates"""
        # In a real implementation, these would be loaded from a file
        # or database with more sophisticated templates

        templates = {
            "viral": [
                "🔥 {trend_name} ist gerade überall! {emoji} Hast du es schon ausprobiert?",
                "Alle reden über {trend_name}! {emoji} Hier ist meine Version! #trending",
                "POV: Du entdeckst {trend_name} zum ersten Mal {emoji} #mustsee",
                "{trend_name} in nur 30 Sekunden erklärt! {emoji} #quicktip"
            ],
            "informative": [
                "3 Gründe, warum {trend_name} gerade viral geht {emoji} #wissen",
                "Das solltest du über {trend_name} wissen! {emoji} #infografik",
                "So funktioniert der {trend_name} Trend {emoji} #tutorial",
                "Die Wahrheit über {trend_name} {emoji} #faktcheck"
            ],
            "humorous": [
                "Wenn {trend_name} nicht so läuft wie geplant {emoji} #fail",
                "POV: Deine Freunde sehen dich beim {trend_name} {emoji} #awkward",
                "Niemand: Absolut niemand: Ich: *macht {trend_name}* {emoji}",
                "{trend_name} expectation vs. reality {emoji} #lol"
            ],
            "motivational": [
                "{trend_name} hat mein Leben verändert! {emoji} #motivation",
                "Wie {trend_name} dir helfen kann, deine Ziele zu erreichen {emoji}",
                "Jeden Tag besser werden mit {trend_name} {emoji} #growth",
                "Der {trend_name} Challenge - Tag 1 {emoji} #30tage"
            ]
        }

        return templates

    async def generate_post_text(
        self,
        trend_data: Dict[str, Any],
        video_id: str,
        style: str = "viral",
        hashtag_count: int = 5,
        include_emojis: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate post text for a social media video

        Args:
            trend_data: Trend data dictionary
            video_id: ID of the video
            style: Post style (e.g., "viral", "informative", "humorous")
            hashtag_count: Number of hashtags to include
            include_emojis: Whether to include emojis

        Returns:
            Dictionary with post text generation results or None if generation fails
        """
        if not trend_data or "id" not in trend_data:
            self.logger.error(
                "Invalid trend data provided for post generation")
            return None

        trend_id = trend_data["id"]
        post_id = str(uuid.uuid4())

        self.logger.info(
            f"Generating post text for trend: {trend_data.get('name', 'Unknown')}")

        try:
            # Generate main text
            main_text = await self._generate_main_text(trend_data, style, include_emojis)

            # Generate hashtags
            hashtags = await self._generate_hashtags(trend_data, hashtag_count)

            # Combine into complete post
            post_text = f"{main_text}\n\n{' '.join(hashtags)}"

            # Create post result
            post_result = {
                "post_id": post_id,
                "trend_id": trend_id,
                "video_id": video_id,
                "main_text": main_text,
                "hashtags": hashtags,
                "complete_text": post_text,
                "style": style,
                "timestamp": datetime.now().isoformat()
            }

            # Cache the result
            self.post_cache[post_id] = post_result

            # Save to disk
            cache_path = os.path.join(self.cache_dir, f"{post_id}.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(post_result, f, ensure_ascii=False, indent=2)

            return post_result

        except Exception as e:
            self.logger.error(f"Error generating post text: {e}")
            return None

    async def _generate_main_text(
        self,
        trend_data: Dict[str, Any],
        style: str,
        include_emojis: bool
    ) -> str:
        """
        Generate main post text

        Args:
            trend_data: Trend data dictionary
            style: Post style
            include_emojis: Whether to include emojis

        Returns:
            Generated main text
        """
        try:
            # Get trend name
            trend_name = trend_data.get("name", "").replace("#", "")

            # Get templates for the selected style
            style_templates = self.templates.get(
                style, self.templates["viral"])

            # Select a random template
            template = random.choice(style_templates)

            # Select an appropriate emoji if needed
            emoji = ""
            if include_emojis:
                emoji = self._select_emoji(style, trend_data)

            # Format the template
            main_text = template.format(trend_name=trend_name, emoji=emoji)

            return main_text

        except Exception as e:
            self.logger.error(f"Error generating main text: {e}")
            return f"Check out this {trend_data.get('name', 'trend')}!"

    def _select_emoji(self, style: str, trend_data: Dict[str, Any]) -> str:
        """Select an appropriate emoji based on style and trend"""
        # Define emoji sets for different styles
        emoji_sets = {
            "viral": ["🔥", "⚡", "💯", "🚀", "👀", "✨", "🤩", "🙌"],
            "informative": ["📊", "📈", "💡", "📝", "🧠", "💭", "📚", "🔍"],
            "humorous": ["😂", "🤣", "😅", "😆", "🙃", "🤪", "😜", "🤦‍♀️"],
            "motivational": ["💪", "🌟", "🏆", "🎯", "✅", "🌈", "🙏", "⭐"]
        }

        # Get emoji set for the selected style
        emoji_set = emoji_sets.get(style, emoji_sets["viral"])

        # Select a random emoji
        return random.choice(emoji_set)

    async def _generate_hashtags(
        self,
        trend_data: Dict[str, Any],
        count: int
    ) -> List[str]:
        """
        Generate hashtags for the post

        Args:
            trend_data: Trend data dictionary
            count: Number of hashtags to generate

        Returns:
            List of hashtags
        """
        try:
            hashtags = []

            # Add trend name as hashtag if it's not already a hashtag
            trend_name = trend_data.get("name", "")
            if not trend_name.startswith("#"):
                clean_name = trend_name.replace(" ", "").replace("#", "")
                if clean_name:
                    hashtags.append(f"#{clean_name}")
            else:
                hashtags.append(trend_name)

            # Add platform as hashtag
            platform = trend_data.get("platform", "").lower()
            if platform:
                hashtags.append(f"#{platform}")

            # Add trend type as hashtag
            trend_type = trend_data.get("type", "").lower()
            if trend_type:
                hashtags.append(f"#{trend_type}")

            # Add generic popular hashtags
            popular_hashtags = [
                "#trending", "#viral", "#fyp", "#foryou", "#foryoupage",
                "#trend", "#challenge", "#deutschland", "#german", "#de",
                "#content", "#socialmedia", "#creator"
            ]

            # Add specific hashtags based on trend data
            if "keywords" in trend_data:
                for keyword in trend_data["keywords"]:
                    clean_keyword = keyword.replace(" ", "").replace("#", "")
                    if clean_keyword:
                        hashtags.append(f"#{clean_keyword}")

            # Fill remaining slots with popular hashtags
            remaining = count - len(hashtags)
            if remaining > 0:
                random.shuffle(popular_hashtags)
                hashtags.extend(popular_hashtags[:remaining])

            # Ensure we don't exceed the requested count
            return hashtags[:count]

        except Exception as e:
            self.logger.error(f"Error generating hashtags: {e}")
            return ["#trend", "#viral", "#content"]

    async def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Get post by ID

        Args:
            post_id: The ID of the post to retrieve

        Returns:
            Post data dictionary or None if not found
        """
        # Check cache first
        if post_id in self.post_cache:
            return self.post_cache[post_id]

        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{post_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                    self.post_cache[post_id] = post_data
                    return post_data
            except Exception as e:
                self.logger.error(f"Error loading post from cache: {e}")

        # Not found
        return None
