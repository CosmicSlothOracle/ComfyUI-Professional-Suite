# Tests

This directory contains tests for the Social Media Video Generation API.

## Running Tests

To run all tests:

```bash
cd social_media_api
pytest
```

To run a specific test file:

```bash
pytest tests/test_trend_service.py
```

To run a specific test:

```bash
pytest tests/test_trend_service.py::test_get_mock_trends
```

## Test Categories

Tests are organized into several categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Functional Tests**: Test complete workflows

## Test Markers

You can use markers to run specific types of tests:

```bash
# Run only asyncio tests
pytest -m asyncio

# Run only slow tests
pytest -m slow

# Run only integration tests
pytest -m integration
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=social_media_api tests/
```

## Writing Tests

When writing tests, follow these guidelines:

1. Each test file should focus on a specific component
2. Use descriptive test names that explain what is being tested
3. Use fixtures to set up test environments
4. Mock external dependencies
5. Test both success and failure cases
6. Test edge cases and boundary conditions

## Test Structure

```
tests/
├── test_trend_service.py - Tests for trend analysis service
├── test_nlp_service.py - Tests for NLP analysis service
├── test_video_service.py - Tests for video generation service
├── test_post_service.py - Tests for post text generation service
└── test_api.py - Tests for API endpoints
```