import os
from typing import Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_prompty import create_chat_prompt

from .base import BasePlanner
from .schemas.plan import Plan
from .utils.tracker import Tracker


class SuperPlanner(BasePlanner):
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
        # Load all prompt templates
        self.system_prompt = create_chat_prompt(
            os.getcwd() + "/prompts/super/system.prompty"
        )
        self.initial_prompt = create_chat_prompt(
            os.getcwd() + "/prompts/super/user_initial.prompty"
        )
        self.replan_prompt = create_chat_prompt(
            os.getcwd() + "/prompts/super/user_replan.prompty"
        )
        self.restructure_prompt = create_chat_prompt(
            os.getcwd() + "/prompts/super/user_restructure.prompty"
        )
        self.history = (
            create_chat_prompt(os.getcwd() + "/prompts/super/system.prompty")
            .invoke({})
            .messages
        )

    def initial_plan(self) -> dict:
        # Combine system + initial
        initial_prompt = self.initial_prompt.invoke(
            {
                "grid_length": self.grid_size,
                "num_agents": self.number_of_agents,
                "num_targets": self.number_of_targets,
                "mission": self.mission_statement,
            }
        ).messages
        self.history += initial_prompt
        ai_message = self.llm.invoke(self.history)
        text_plan = ai_message.content
        print(text_plan)
        self.history.append(ai_message)
        hla_plan = self.llm.with_structured_output(Plan).invoke(
            self.history + self.restructure_prompt.invoke({}).messages,
        )
        print(hla_plan)
        # Always restructure into JSON HLAs
        return text_plan, hla_plan.agents

    # def initial_plan(self) -> dict:
    #     # Generate a textual plan
    #     prompt = create_chat_prompt(
    #         os.getcwd() + "/prompts/initial_text_planner.prompty"
    #     )
    #     initial_text_planner = prompt | self.llm
    #     text_plan = initial_text_planner.invoke(
    #         {
    #             "grid_length": self.grid_size,
    #             "num_agents": self.number_of_agents,
    #             "num_targets": self.number_of_targets,
    #             "mission": self.mission_statement,
    #         }
    #     ).content
    #     print(text_plan)
    #     return self.restructure_text_plan(text_plan)

    def replan(self, agents, observations, rewards, terminations, truncations, infos):
        del observations["global"]
        for k, v in observations.items():
            location = tuple(int(x) for x in v["location"])
            self.agent_trajectories[k].append(location)
        # stuck = False
        # for k, v in self.agent_trajectories.items():
        #     if v[-1] == v[-2]:
        #         stuck = True
        #         break

        if (
            any([r == 1 for r in rewards.values()])
            or any(agents.idle(i) for i in range(self.number_of_agents))
            # or stuck
        ):
            reason = ""
            if any([r == 1 for r in rewards.values()]):
                found_targets_agents = [k for k, v in rewards.items() if v == 1]
                found_targets_locations = [
                    tuple(int(x) for x in observations[i]["location"])
                    for i in found_targets_agents
                ]
                for i, target in zip(found_targets_agents, found_targets_locations):
                    reason += f"Agent {i} has found the target at {target}\n"

            elif any(agents.idle(i) for i in range(self.number_of_agents)):
                idle_agents = [
                    i for i in range(self.number_of_agents) if agents.idle(i)
                ]
                reason = f"The following agents are idle: {str(idle_agents)}"
            elif stuck:
                print("Re-planning due to stuck")
            print(f"Re-planning due to: {reason}")

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

            truncated_history = trim_messages(
                self.history,
                include_system=True,
                max_tokens=20000,
                token_counter=count_tokens_approximately,
            )
            replan_prompt = self.replan_prompt.invoke(
                {
                    "reason": reason,
                    "targets_found": self.found_targets,
                    "agent_locations": agent_locations,
                }
            ).messages
            print(replan_prompt)
            truncated_history += replan_prompt
            ai_message = self.llm.invoke(self.history)
            text_plan = ai_message.content
            print(text_plan)
            self.history.append(ai_message)
            truncated_history.append(ai_message)
            hla_plan = self.llm.with_structured_output(Plan).invoke(
                truncated_history + self.restructure_prompt.invoke({}).messages
            )
            print(hla_plan)
            # Always restructure into JSON HLAs
            return text_plan, hla_plan.agents

        self.tracker.observe(observations, rewards)
        return "", {}

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
