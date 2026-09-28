from fastapi import FastAPI, HTTPException

from app.agent.budget import SearchBudget
from app.agent.evaluator import EvidenceEvaluator
from app.agent.research_executor import ResearchExecutor
from app.agent.research_loop import ResearchLoop
from app.agent.strategy_engine import StrategyEngine
from app.agent.strategy_selector import StrategySelector
from app.agent.strategy_switcher import StrategySwitcher
from app.agent.synthesizer import FinalSynthesizer
from app.api.models import MissionRequest, MissionResponse
from app.llm.factory import get_llm_provider
from app.tools.serpapi import SerpApiTools


app = FastAPI(title="MISSIONCTRL API", version="1.0.0")


def build_research_loop(search_budget: int) -> ResearchLoop:
    """Build independent mission-scoped dependencies for one API request."""

    llm = get_llm_provider()
    executor = ResearchExecutor(
        serpapi_tools=SerpApiTools(),
        llm=llm,
        budget=SearchBudget(maximum=search_budget),
    )
    return ResearchLoop(
        executor=executor,
        strategy_engine=StrategyEngine(llm),
        strategy_selector=StrategySelector(llm),
        evaluator=EvidenceEvaluator(),
        strategy_switcher=StrategySwitcher(llm),
        synthesizer=FinalSynthesizer(llm),
    )


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/api/v1/missions", response_model=MissionResponse)
async def create_mission(request: MissionRequest) -> MissionResponse:
    try:
        research_loop = build_research_loop(request.search_budget)
        result = await research_loop.run(
            mission=request.mission,
            plan=request.plan,
        )
    except (ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=500,
            detail="The research mission could not be completed.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred.",
        ) from error

    return MissionResponse.model_validate(result.model_dump())
