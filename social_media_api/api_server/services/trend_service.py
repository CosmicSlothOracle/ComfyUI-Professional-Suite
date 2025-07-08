"""
Trend Analysis Service
Handles fetching and analyzing social media trends from various platforms
"""

import logging
import aiohttp
import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
from bs4 import BeautifulSoup
import pandas as pd


class TrendAnalysisService:
    """Service for analyzing social media trends"""

    def __init__(self):
        """Initialize the trend analysis service"""
        self.logger = logging.getLogger(__name__)
        self.trends_cache = {}
        self.cache_dir = "social_media_api/temp/trends_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load any cached trends
        self._load_cached_trends()

    def _load_cached_trends(self):
        """Load cached trends from disk"""
        try:
            cache_files = os.listdir(self.cache_dir)
            for file in cache_files:
                if file.endswith(".json"):
                    with open(os.path.join(self.cache_dir, file), 'r', encoding='utf-8') as f:
                        trend_data = json.load(f)
                        if "id" in trend_data:
                            self.trends_cache[trend_data["id"]] = trend_data

            self.logger.info(f"Loaded {len(self.trends_cache)} cached trends")
        except Exception as e:
            self.logger.error(f"Error loading cached trends: {e}")

    async def analyze_trends(
        self,
        platforms: List[str],
        region: str,
        age_range: List[int],
        limit: int = 10,
        include_hashtags: bool = True,
        include_sounds: bool = True,
        include_formats: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze trends from specified social media platforms

        Args:
            platforms: List of platforms to analyze (e.g., "tiktok", "instagram")
            region: Region code (e.g., "DE" for Germany)
            age_range: Target age range [min, max]
            limit: Maximum number of trends to return
            include_hashtags: Whether to include hashtag trends
            include_sounds: Whether to include sound/music trends
            include_formats: Whether to include format/template trends

        Returns:
            List of trend data dictionaries
        """
        self.logger.info(
            f"Analyzing trends for platforms: {platforms}, region: {region}")

        all_trends = []

        try:
            # Process each platform
            for platform in platforms:
                platform_trends = await self._fetch_platform_trends(
                    platform=platform,
                    region=region,
                    age_range=age_range,
                    include_hashtags=include_hashtags,
                    include_sounds=include_sounds,
                    include_formats=include_formats
                )

                if platform_trends:
                    all_trends.extend(platform_trends)
                else:
                    self.logger.warning(
                        f"No trends found for platform: {platform}")

            # Sort trends by popularity score
            all_trends.sort(key=lambda x: x.get(
                "popularity_score", 0), reverse=True)

            # Limit results
            limited_trends = all_trends[:limit]

            # Cache trends
            for trend in limited_trends:
                if "id" in trend:
                    self.trends_cache[trend["id"]] = trend
                    # Save to disk
                    with open(os.path.join(self.cache_dir, f"{trend['id']}.json"), 'w', encoding='utf-8') as f:
                        json.dump(trend, f, ensure_ascii=False, indent=2)

            self.logger.info(
                f"Found {len(limited_trends)} trends across {len(platforms)} platforms")
            return limited_trends

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return []

    async def _fetch_platform_trends(
        self,
        platform: str,
        region: str,
        age_range: List[int],
        include_hashtags: bool = True,
        include_sounds: bool = True,
        include_formats: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch trends from a specific platform

        Args:
            platform: Platform name (e.g., "tiktok", "instagram")
            region: Region code
            age_range: Target age range [min, max]
            include_hashtags: Whether to include hashtag trends
            include_sounds: Whether to include sound/music trends
            include_formats: Whether to include format/template trends

        Returns:
            List of trend data dictionaries for the platform
        """
        if platform.lower() == "tiktok":
            return await self._fetch_tiktok_trends(region, age_range, include_hashtags, include_sounds, include_formats)
        elif platform.lower() == "instagram":
            return await self._fetch_instagram_trends(region, age_range, include_hashtags, include_sounds, include_formats)
        else:
            self.logger.warning(f"Unsupported platform: {platform}")
            return []

    async def _fetch_tiktok_trends(
        self,
        region: str,
        age_range: List[int],
        include_hashtags: bool,
        include_sounds: bool,
        include_formats: bool
    ) -> List[Dict[str, Any]]:
        """
        Fetch TikTok trends

        Note: In a production environment, this would use the TikTok API or a dedicated
        scraping service. For this demonstration, we'll implement a basic scraper
        with appropriate error handling and empty result detection.
        """
        self.logger.info(f"Fetching TikTok trends for region: {region}")
        trends = []

        try:
            async with aiohttp.ClientSession() as session:
                # For demonstration purposes - in production would use proper API endpoints
                # This is a placeholder for the actual implementation
                if include_hashtags:
                    url = f"https://www.tiktok.com/discover?lang={region.lower()}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            hashtag_trends = self._parse_tiktok_hashtags(
                                html, age_range)
                            if hashtag_trends:
                                trends.extend(hashtag_trends)
                            else:
                                self.logger.warning(
                                    "No hashtag trends found on TikTok")
                        else:
                            self.logger.error(
                                f"Failed to fetch TikTok hashtags: {response.status}")

                # Similar implementations would be added for sounds and formats

                # If no trends were found through scraping, return empty list
                # IMPORTANT: We don't generate fake data if the API fails
                if not trends:
                    self.logger.warning("No TikTok trends found")
                    return []

                return trends

        except Exception as e:
            self.logger.error(f"Error fetching TikTok trends: {e}")
            return []  # Return empty list on error, don't generate fake data

    def _parse_tiktok_hashtags(self, html: str, age_range: List[int]) -> List[Dict[str, Any]]:
        """Parse TikTok hashtag trends from HTML"""
        # This is a placeholder for actual HTML parsing logic
        # In a real implementation, this would extract hashtags from the HTML

        # IMPORTANT: Return empty list if no trends are found
        # Don't generate fake data if the scraping fails
        return []

    async def _fetch_instagram_trends(
        self,
        region: str,
        age_range: List[int],
        include_hashtags: bool,
        include_sounds: bool,
        include_formats: bool
    ) -> List[Dict[str, Any]]:
        """
        Fetch Instagram trends

        Similar to TikTok implementation, this would use the Instagram API
        or a dedicated scraping service in a production environment.
        """
        self.logger.info(f"Fetching Instagram trends for region: {region}")
        trends = []

        try:
            async with aiohttp.ClientSession() as session:
                # For demonstration purposes - in production would use proper API endpoints
                # This is a placeholder for the actual implementation
                if include_hashtags:
                    url = f"https://www.instagram.com/explore/tags/trending/"
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            hashtag_trends = self._parse_instagram_hashtags(
                                html, age_range)
                            if hashtag_trends:
                                trends.extend(hashtag_trends)
                            else:
                                self.logger.warning(
                                    "No hashtag trends found on Instagram")
                        else:
                            self.logger.error(
                                f"Failed to fetch Instagram hashtags: {response.status}")

                # Similar implementations would be added for sounds and formats

                # If no trends were found through scraping, return empty list
                # IMPORTANT: We don't generate fake data if the API fails
                if not trends:
                    self.logger.warning("No Instagram trends found")
                    return []

                return trends

        except Exception as e:
            self.logger.error(f"Error fetching Instagram trends: {e}")
            return []  # Return empty list on error, don't generate fake data

    def _parse_instagram_hashtags(self, html: str, age_range: List[int]) -> List[Dict[str, Any]]:
        """Parse Instagram hashtag trends from HTML"""
        # This is a placeholder for actual HTML parsing logic
        # In a real implementation, this would extract hashtags from the HTML

        # IMPORTANT: Return empty list if no trends are found
        # Don't generate fake data if the scraping fails
        return []

    async def get_trend_by_id(self, trend_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trend by its ID

        Args:
            trend_id: The ID of the trend to retrieve

        Returns:
            Trend data dictionary or None if not found
        """
        # Check cache first
        if trend_id in self.trends_cache:
            return self.trends_cache[trend_id]

        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{trend_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    trend_data = json.load(f)
                    self.trends_cache[trend_id] = trend_data
                    return trend_data
            except Exception as e:
                self.logger.error(f"Error loading trend from cache: {e}")

        # Not found
        return None

    async def get_mock_trends_for_testing(self) -> List[Dict[str, Any]]:
        """
        Get mock trends for testing purposes ONLY

        This should ONLY be used in development/testing environments,
        not in production.

        Returns:
            List of mock trend data dictionaries
        """
        self.logger.warning(
            "Using mock trends for testing - NOT FOR PRODUCTION USE")

        mock_trends = []

        # TikTok mock trends
        tiktok_trend_1 = {
            "id": str(uuid.uuid4()),
            "platform": "tiktok",
            "type": "hashtag",
            "name": "#DanceChallenge2023",
            "description": "Users dancing to the latest hit song",
            "popularity_score": 95,
            "views": 15000000,
            "videos_count": 25000,
            "avg_engagement_rate": 8.3,
            "target_demographics": {
                "age_min": 16,
                "age_max": 24,
                "gender_distribution": {"female": 65, "male": 35}
            },
            "examples": [
                {"url": "https://example.com/video1", "views": 2500000},
                {"url": "https://example.com/video2", "views": 1800000}
            ],
            "created_at": datetime.now().isoformat()
        }

        tiktok_trend_2 = {
            "id": str(uuid.uuid4()),
            "platform": "tiktok",
            "type": "sound",
            "name": "Viral Song Remix",
            "description": "Remix of a popular song used in various creative videos",
            "popularity_score": 88,
            "views": 12000000,
            "videos_count": 18000,
            "avg_engagement_rate": 7.5,
            "target_demographics": {
                "age_min": 18,
                "age_max": 27,
                "gender_distribution": {"female": 55, "male": 45}
            },
            "examples": [
                {"url": "https://example.com/video3", "views": 1900000},
                {"url": "https://example.com/video4", "views": 1500000}
            ],
            "created_at": datetime.now().isoformat()
        }

        # Instagram mock trends
        instagram_trend_1 = {
            "id": str(uuid.uuid4()),
            "platform": "instagram",
            "type": "hashtag",
            "name": "#PhotoChallenge2023",
            "description": "Creative photo challenge gaining popularity",
            "popularity_score": 92,
            "posts_count": 35000,
            "avg_likes": 15000,
            "avg_engagement_rate": 6.8,
            "target_demographics": {
                "age_min": 17,
                "age_max": 26,
                "gender_distribution": {"female": 60, "male": 40}
            },
            "examples": [
                {"url": "https://example.com/post1", "likes": 250000},
                {"url": "https://example.com/post2", "likes": 180000}
            ],
            "created_at": datetime.now().isoformat()
        }

        instagram_trend_2 = {
            "id": str(uuid.uuid4()),
            "platform": "instagram",
            "type": "format",
            "name": "Transition Reels",
            "description": "Creative transitions between scenes in short videos",
            "popularity_score": 85,
            "posts_count": 28000,
            "avg_likes": 12000,
            "avg_engagement_rate": 5.9,
            "target_demographics": {
                "age_min": 16,
                "age_max": 25,
                "gender_distribution": {"female": 58, "male": 42}
            },
            "examples": [
                {"url": "https://example.com/reel1", "likes": 220000},
                {"url": "https://example.com/reel2", "likes": 175000}
            ],
            "created_at": datetime.now().isoformat()
        }

        mock_trends.extend([tiktok_trend_1, tiktok_trend_2,
                           instagram_trend_1, instagram_trend_2])

        # Cache these mock trends
        for trend in mock_trends:
            self.trends_cache[trend["id"]] = trend

        return mock_trends
