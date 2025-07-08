"""
Tests for the NLP analysis service
"""

from api_server.services.nlp_service import NLPService
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def nlp_service():
    """Create an NLP service instance for testing"""
    return NLPService()


def test_nlp_service_initialization(nlp_service):
    """Test that the NLP service initializes correctly"""
    assert nlp_service is not None
    assert hasattr(nlp_service, "analysis_cache")
    assert isinstance(nlp_service.analysis_cache, dict)


@pytest.mark.asyncio
async def test_basic_analysis_fallback(nlp_service):
    """Test that _perform_basic_analysis creates a valid analysis"""
    # Create a simple trend
    trend_data = {
        "id": "test-trend-id",
        "name": "#TestTrend",
        "platform": "tiktok",
        "type": "hashtag"
    }

    # Perform basic analysis
    analysis = await nlp_service._perform_basic_analysis(trend_data)

    # Check analysis structure
    assert analysis is not None
    assert "trend_id" in analysis
    assert "analysis_id" in analysis
    assert "sentiment" in analysis
    assert "classification" in analysis
    assert "content_mechanics" in analysis
    assert "keywords" in analysis
    assert "summary" in analysis

    # Check specific values
    assert analysis["trend_id"] == "test-trend-id"
    assert analysis["classification"]["primary_type"] == "hashtag"
    assert "TestTrend" in analysis["keywords"]


@pytest.mark.asyncio
async def test_analyze_trend_invalid_input(nlp_service):
    """Test that analyze_trend handles invalid input correctly"""
    # Test with None
    result = await nlp_service.analyze_trend(None)
    assert result is None

    # Test with empty dict
    result = await nlp_service.analyze_trend({})
    assert result is None

    # Test with missing ID
    result = await nlp_service.analyze_trend({"name": "Test"})
    assert result is None


@pytest.mark.asyncio
async def test_analyze_trend_caching(nlp_service):
    """Test that analyze_trend caches results correctly"""
    # Create a simple trend
    trend_data = {
        "id": "cache-test-id",
        "name": "#CacheTest",
        "platform": "tiktok",
        "type": "hashtag"
    }

    # First analysis should not be cached
    assert "cache-test-id" not in nlp_service.analysis_cache

    # Perform analysis
    analysis1 = await nlp_service.analyze_trend(trend_data)
    assert analysis1 is not None

    # Check that result was cached
    assert "cache-test-id" in nlp_service.analysis_cache

    # Perform analysis again
    analysis2 = await nlp_service.analyze_trend(trend_data)

    # Check that we got the same object (from cache)
    assert analysis2 is not None
    assert analysis1["analysis_id"] == analysis2["analysis_id"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
