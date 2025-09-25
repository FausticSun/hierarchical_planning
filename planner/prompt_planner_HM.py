import os
from typing import Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_prompty import create_chat_prompt

from .base import BasePlanner
from .schemas.plan import Plan
from .utils.tracker import Tracker

import time

class PromptPlanner(BasePlanner):
    llm: BaseChatModel
    tracker: Tracker
    mission_statement: str = ""
    number_of_agents: int = -1
    grid_size: int = -1
    number_of_targets: int = -1
    agent_positions: Dict[int, Tuple[int, int]]

    def __init__(
        self,
        llm: BaseChatModel,
        grid_size,
        observations,
        infos,
    ) -> None:
        self.llm = llm
        self.mission_statement = observations[0]["mission"]
        self.number_of_agents = len(observations.keys()) - 1
        self.number_of_targets = observations["global"]["num_goals"]
        self.agent_positions = {i: (1, 1) for i in range(0, self.number_of_agents)}
        self.grid_size = grid_size
        self.tracker = Tracker(grid_size)

    def initial_plan(self) -> dict:
        # Generate a textual plan
        prompt = create_chat_prompt(
            os.getcwd() + "/prompts/initial_text_planner.prompty"
        )
        initial_text_planner = prompt | self.llm
        text_plan = initial_text_planner.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "num_targets": self.number_of_targets,
                "mission": self.mission_statement,
            }
        ).content
        print(text_plan)

        # Convert the textual plan into structured instructions
        prompt = create_chat_prompt(os.getcwd() + "/prompts/plan_structurer.prompty")
        model_with_structure = self.llm.with_structured_output(Plan)
        plan_structurer = prompt | model_with_structure
        plan = plan_structurer.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "num_targets": self.number_of_targets,
                "mission": self.mission_statement,
                "plan": text_plan,
            }
        )
        return plan.agents

    def replan(
        self, agents, observations, rewards, terminations, truncations, infos
    ) -> dict:
        del observations["global"]

        found = False
        if any([r == 1 for r in rewards.values()]):
            found = True
            agent_id = [id for id, r in rewards.items() if r == 1][0]
            print(f"agent_id: {agent_id} found target!")
            print(observations[agent_id]["location"])
            # Re-plan when a target is found
            # Add new mission
            # Current mission statement:

            curr_mission = observations[0]['mission']
            print(curr_mission)
            return {}
        self.tracker.observe(observations, rewards)
        return {}
