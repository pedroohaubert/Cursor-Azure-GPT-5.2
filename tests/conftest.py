"""Defines fixtures available to all tests."""

import logging

import pytest
from flask import Flask
from webtest import TestApp

from app import create_app
from app.registry.model_config import ModelConfig


@pytest.fixture
def app() -> Flask:
    """Create application for the tests."""
    _app = create_app("tests.settings")
    _app.logger.setLevel(logging.CRITICAL)
    ctx = _app.test_request_context()
    ctx.push()

    yield _app

    ctx.pop()


@pytest.fixture
def testapp(app) -> TestApp:
    """Create Webtest app."""
    return TestApp(app)


def create_test_azure_config(reasoning_effort="high"):
    """Create test Azure model configuration."""
    return ModelConfig(
        name=f"test-gpt-{reasoning_effort}",
        backend="azure",
        api_model="gpt-5",
        reasoning_effort=reasoning_effort,
        summary_level="detailed",
        verbosity_level="medium",
    )
