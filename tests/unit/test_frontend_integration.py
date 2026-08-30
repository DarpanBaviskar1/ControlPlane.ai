"""Frontend and Static Serving integration tests."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_serve_dashboard_root():
    """GET / returns HTML dashboard with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ControlPlane" in response.text
    assert "Enterprise AI Proxy Gateway" in response.text


def test_serve_dashboard_alias():
    """GET /dashboard returns HTML dashboard with 200 OK."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Interactive Studio" in response.text


def test_serve_static_css():
    """GET /static/css/style.css returns CSS with 200 OK."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "--bg-primary" in response.text


def test_serve_static_js():
    """GET /static/js/app.js returns JS with 200 OK."""
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "PRESETS" in response.text
    assert "executeStreamingRequest" in response.text


def test_get_profiles_endpoint():
    """GET /v1/profiles returns loaded use case profiles."""
    response = client.get("/v1/profiles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "customer_chatbot" in data
    assert "internal_copilot" in data
