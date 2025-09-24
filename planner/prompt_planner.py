import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_prompty import create_chat_prompt

from .base import BasePlanner
from .schemas.plan import Plan
from .utils.tracker import Tracker


class PromptPlanner(BasePlanner):
    llm: BaseChatModel
    tracker: Tracker
    mission_statement: str = ""
    number_of_agents: int = -1
    grid_size: int = -1
    number_of_targets: int = -1

    def __init__(
        self,
        llm: BaseChatModel,
        observations,
        infos,
    ) -> None:
        self.llm = llm
        self.mission_statement = observations[0]["mission"]
        self.number_of_agents = len(observations.keys()) - 1
        self.number_of_targets = observations["global"]["num_goals"]

    def initial_plan(self):
        model_with_structure = self.llm.with_structured_output(Plan)
        prompt = create_chat_prompt(os.getcwd() + "/prompts/initial_planner.prompty")
        initial_planner = prompt | model_with_structure
        plan = initial_planner.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "mission": self.mission_statement,
            }
        )
        return plan

    def replan(self, observations, rewards, terminations, truncations, infos):
        self.tracker.observe(observations, rewards)
        pass
