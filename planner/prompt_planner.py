import os
from typing import Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_prompty import create_chat_prompt

from .base import BasePlanner
from .schemas.plan import Plan, StopAction
from .utils.tracker import Tracker


class PromptPlanner(BasePlanner):
    llm: BaseChatModel
    tracker: Tracker
    mission_statement: str = ""
    number_of_agents: int = -1
    grid_size: int = -1
    number_of_targets: int = -1
    agent_trajectories: Dict[int, List[Tuple[int, int]]]
    found_targets = set()

    def __init__(
        self,
        llm: BaseChatModel,
        grid_size,
        observations,
        infos,
    ) -> None:
        self.llm = llm
        self.mission_statement = str(observations[0]["mission"])
        self.number_of_agents = len(observations.keys()) - 1
        self.number_of_targets = observations["global"]["num_goals"]
        self.agent_trajectories = {i: [(1, 1)] for i in range(0, self.number_of_agents)}
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
        return self.restructure_text_plan(text_plan)

    def replan(
        self, agents, observations, rewards, terminations, truncations, infos
    ) -> dict:
        del observations["global"]
        for k, v in observations.items():
            location = tuple(int(x) for x in v["location"])
            self.agent_trajectories[k].append(location)
        stuck = False
        for k, v in self.agent_trajectories.items():
            if v[-1] == v[-2]:
                stuck = True
                break

        if (
            any([r == 1 for r in rewards.values()])
            or any(agents.idle(i) for i in range(self.number_of_agents))
            or stuck
        ):
            if any([r == 1 for r in rewards.values()]):
                print("Re-planning due to found target")
            if any(agents.idle(i) for i in range(self.number_of_agents)):
                print("Re-planning due to agent idle")
            if stuck:
                print("Re-planning due to stuck")

            # Re-plan when a target is found
            found_targets_agents = [k for k, v in rewards.items() if v == 1]
            found_targets_locations = [
                tuple(int(x) for x in observations[i]["location"])
                for i in found_targets_agents
            ]
            self.found_targets.update(found_targets_locations)
            agent_locations = {
                k: tuple(int(x) for x in v["location"]) for k, v in observations.items()
            }
            # Generate a textual plan
            prompt = create_chat_prompt(
                os.getcwd() + "/prompts/replan_text_planner.prompty"
            )
            replan_text_planner = prompt | self.llm
            text_plan = replan_text_planner.invoke(
                {
                    "grid_length": self.grid_size,
                    "num_agents": self.number_of_agents,
                    "num_targets": self.number_of_targets,
                    "mission": self.mission_statement,
                    "targets_found": str(self.found_targets),
                    "agent_locations": str(agent_locations),
                }
            ).content
            print(text_plan)
            # Convert the textual plan into structured instructions
            new_plan = self.restructure_text_plan(text_plan)
            for k in new_plan:
                new_plan[k].insert(0, StopAction())
            return new_plan

        self.tracker.observe(observations, rewards)
        return {}

    def restructure_text_plan(self, text_plan) -> dict:
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
