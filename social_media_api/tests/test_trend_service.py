"""
Tests for the trend analysis service
"""

from api_server.services.trend_service import TrendAnalysisService
import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def trend_service():
    """Create a trend service instance for testing"""
    return TrendAnalysisService()


def test_trend_service_initialization(trend_service):
    """Test that the trend service initializes correctly"""
    assert trend_service is not None
    assert hasattr(trend_service, "trends_cache")
    assert isinstance(trend_service.trends_cache, dict)


@pytest.mark.asyncio
async def test_get_mock_trends(trend_service):
    """Test that the get_mock_trends_for_testing method returns valid data"""
    trends = await trend_service.get_mock_trends_for_testing()

    # Check that we got trends
    assert trends is not None
    assert len(trends) > 0

    # Check trend structure
    trend = trends[0]
    assert "id" in trend
    assert "platform" in trend
    assert "name" in trend
    assert "popularity_score" in trend

    # Check trend values
    assert isinstance(trend["id"], str)
    assert trend["platform"] in ["tiktok", "instagram"]
    assert isinstance(trend["name"], str)
    assert isinstance(trend["popularity_score"], (int, float))


@pytest.mark.asyncio
async def test_analyze_trends_empty_result():
    """Test that analyze_trends handles empty results correctly"""
    trend_service = TrendAnalysisService()

    # Mock the _fetch_platform_trends method to return empty results
    with patch.object(trend_service, "_fetch_platform_trends", return_value=asyncio.Future()) as mock_fetch:
        mock_fetch.return_value.set_result([])

        # Call analyze_trends
        result = await trend_service.analyze_trends(
            platforms=["tiktok"],
            region="DE",
            age_range=[16, 27]
        )

        # Check that the method was called with correct parameters
        mock_fetch.assert_called_once_with(
            platform="tiktok",
            region="DE",
            age_range=[16, 27],
            include_hashtags=True,
            include_sounds=True,
            include_formats=True
        )

        # Check that we got an empty list
        assert result == []


@pytest.mark.asyncio
async def test_get_trend_by_id(trend_service):
    """Test that get_trend_by_id returns the correct trend"""
    # Get mock trends
    trends = await trend_service.get_mock_trends_for_testing()
    trend_id = trends[0]["id"]

    # Get trend by ID
    trend = await trend_service.get_trend_by_id(trend_id)

    # Check that we got the right trend
    assert trend is not None
    assert trend["id"] == trend_id


@pytest.mark.asyncio
async def test_get_trend_by_id_not_found(trend_service):
    """Test that get_trend_by_id returns None for non-existent trends"""
    trend = await trend_service.get_trend_by_id("non-existent-id")
    assert trend is None


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
