"""API integration tests against a fake Redis (no network needed)."""

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app


@pytest.fixture()
def client(monkeypatch):
    fake = fakeredis.aioredis.FakeRedis()

    def _fake_client(settings=None):
        return fake

    async def _fake_ping(client=None):
        return True

    monkeypatch.setattr(routes, "get_redis_client", _fake_client)
    monkeypatch.setattr(routes, "redis_ping", _fake_ping)
    return TestClient(app)


def test_submit_and_poll_run(client):
    r = client.post("/research", json={
        "question": "What are vector databases?",
        "max_subquestions": 2,
        "max_searches": 4,
    })
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    assert run_id

    g = client.get(f"/research/{run_id}")
    assert g.status_code == 200
    body = g.json()
    assert body["run_id"] == run_id
    assert body["status"] == "QUEUED"
    assert body["question"] == "What are vector databases?"
    assert body["progress"]["subquestions_answered"] == 0

    t = client.get(f"/research/{run_id}/trace")
    assert t.status_code == 200
    assert t.json()["steps"] == []


def test_run_not_found(client):
    assert client.get("/research/nope").status_code == 404
    assert client.get("/research/nope/trace").status_code == 404


def test_health_and_ready(client):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    rd = client.get("/ready")
    assert rd.status_code == 200
    assert rd.json()["ready"] is True
