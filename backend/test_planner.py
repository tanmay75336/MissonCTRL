import asyncio
import json

from app.agent.planner import MissionPlanner
from app.llm.factory import get_llm_provider


async def main():

    mission = (
        "I have ₹2 lakh and want to find a "
        "student-focused business opportunity in Mumbai."
    )

    llm = get_llm_provider("groq")

    planner = MissionPlanner(llm)

    plan = await planner.create_plan(mission)

    print("\n" + "=" * 60)
    print("MISSIONCTRL — MISSION PLAN")
    print("=" * 60)

    print(
        json.dumps(
            plan.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())