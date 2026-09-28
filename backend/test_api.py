from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agent.evaluator import NextActionDecision
from app.agent.events import ResearchEvent
from app.agent.research_loop import ResearchLoopResult
from app.main import app


class _FakeLoop:
    async def run(self, mission: str, plan: str = "") -> ResearchLoopResult:
        return ResearchLoopResult(
            mission=mission,
            answer="A mocked evidence-grounded answer.",
            facts=["A supported fact."],
            unknowns=["An unresolved item."],
            searches_used=1,
            remaining_searches=3,
            decisions=[
                NextActionDecision(
                    status="SUFFICIENT",
                    reason="Mock evidence is sufficient.",
                    next_action="SYNTHESIZE",
                    confidence=0.9,
                )
            ],
            events=[
                ResearchEvent(
                    type="MISSION_STARTED",
                    message="Mission started.",
                ),
                ResearchEvent(
                    type="DECISION",
                    message="Evidence sufficiency decision recorded.",
                    data={"next_action": "SYNTHESIZE"},
                ),
                ResearchEvent(
                    type="MISSION_COMPLETED",
                    message="Mission completed.",
                ),
            ],
        )


def test_create_mission_returns_structured_result() -> None:
    with patch("app.main.build_research_loop", return_value=_FakeLoop()) as builder:
        response = TestClient(app).post(
            "/api/v1/missions",
            json={
                "mission": "Find official hackathon requirements.",
                "plan": "Use official sources first.",
                "search_budget": 4,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["facts"]
    assert data["unknowns"]
    assert "sources" in data
    assert data["searches_used"] == 1
    assert data["remaining_searches"] == 3
    assert data["decisions"]
    assert {event["type"] for event in data["events"]} >= {
        "MISSION_STARTED", "DECISION", "MISSION_COMPLETED"
    }
    builder.assert_called_once_with(4)


def test_create_mission_rejects_blank_mission() -> None:
    response = TestClient(app).post(
        "/api/v1/missions",
        json={"mission": "   "},
    )

    assert response.status_code == 422


def test_create_mission_rejects_invalid_budget() -> None:
    response = TestClient(app).post(
        "/api/v1/missions",
        json={"mission": "Find official information.", "search_budget": 9},
    )

    assert response.status_code == 422


def test_health_and_ready_endpoints() -> None:
    client = TestClient(app)

    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/ready").json() == {"status": "ready"}


def test_normal_research_stop_returns_a_valid_response() -> None:
    class _StoppedLoop:
        async def run(self, mission: str, plan: str = "") -> ResearchLoopResult:
            return ResearchLoopResult(
                mission=mission,
                answer="The evidence is incomplete.",
                unknowns=["A specific gap remains."],
                unresolved_questions=["No new specific research question could be generated."],
                searches_used=1,
                remaining_searches=3,
                decisions=[
                    NextActionDecision(
                        status="INSUFFICIENT",
                        reason="No unique question remains.",
                        next_action="STOP",
                        confidence=1.0,
                    )
                ],
            )

    with patch("app.main.build_research_loop", return_value=_StoppedLoop()):
        response = TestClient(app).post(
            "/api/v1/missions",
            json={"mission": "Find official information."},
        )

    assert response.status_code == 200
    assert response.json()["decisions"][-1]["next_action"] == "STOP"
