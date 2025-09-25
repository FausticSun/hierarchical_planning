# type: ignore
import configparser
import logging
import os
import sys

import gymnasium as gym
import imageio
import pandas as pd

import multigrid.envs
from agents import AgentCollection
from models import local_llm as llm

#####################################################
# TODO: Import helper functions and classes, if any.
from planner import SuperPlanner as Planner

#####################################################
section = "env.9"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"out_{section}.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
os.makedirs("gif", exist_ok=True)

# Load test environment configurations
config = configparser.ConfigParser()
config.read("env_config_test.ini")

results = []
M = config.getint(section, "number_of_agents")
N = config.getint(section, "grid_size")
goals = eval(config.get(section, "goals"))
mission_statement = eval(config.get(section, "mission_statement"))
num_trials = config.getint(section, "number_of_trials")

# Print the environment configuration
logging.info(f"Env: {section}")
logging.info(f"Number of agents: {M}")
logging.info(f"Grid size: {N}")
logging.info(f"Mission statement: {mission_statement}")
logging.info(f"Total number of targets: {len(goals)}")

results_i = []
for trial in range(1, num_trials + 1):
    logging.info(f"Trial {trial}/{num_trials} for {section}")
    frames = []

    try:
        # Create the environment with specified parameters
        env = multigrid.envs.EmptyEnvV2(
            size=N,  # Specify the size of the grid, N
            agents=M,  # Specify number of agents, M
            goals=goals,  # Specify target positions for agents
            mission_space=mission_statement,  # Mission statement for the environment
            render_mode="rgb_array",
            hidden_goals=True,
            max_steps=N * N,  # For debugging only
        )
        observations, infos = env.reset()
        frames.append(env.render())
        agents = AgentCollection(num=M)

        #####################################################
        # TODO: Add code to create the initial HLA plan when needed.
        planner = Planner(
            llm=llm,
            grid_size=N,
            observations=observations,
            infos=infos,
        )
        text_plan, plan = planner.initial_plan()
        for agent, actions in plan.items():
            for action in actions:
                agents.tell({agent: action.serialize()})

        #####################################################

        while not env.unwrapped.is_done():
            a = agents.act()
            observations, rewards, terminations, truncations, infos = env.step(a)
            frames.append(env.render())
            #####################################################
            # TODO: Add code to support dynamic replanning when necessary.
            text_plan, plan = planner.replan(
                agents, observations, rewards, terminations, truncations, infos
            )
            for agent, actions in plan.items():
                agents.tell({agent: "stop()"})
                for action in actions:
                    agents.tell({agent: action.serialize()})
            #####################################################

        logging.info(f"Number of steps taken: {env.unwrapped.step_count}")
        logging.info(f"Rewards: {infos['total_reward']}")
        num_targets_left = len(env.unwrapped.goals)
        logging.info(
            f"Number of targets found: {env.unwrapped.total_goals - num_targets_left}"
        )
        logging.info(f"Number of targets remaining: {num_targets_left}")
        imageio.mimsave(f"gif/{section}_{trial}.gif", frames, fps=30, loop=0)

        total_targets, steps_taken = (
            env.unwrapped.total_goals,
            env.unwrapped.step_count,
        )
        final_reward, targets_found = (
            infos["total_reward"],
            env.unwrapped.total_goals - num_targets_left,
        )

    except Exception as e:
        logging.error(f"An error occurred during trial {trial} for {section}: {e}")
        num_targets_left = None
        total_targets, steps_taken, final_reward, targets_found = (
            None,
            None,
            None,
            None,
        )

    finally:
        results_i = {
            "env_name": section,
            "trial_id": trial,
            "num_agents": M,
            "gridsize": N,
            "total_targets": total_targets,
            "steps_taken": steps_taken,
            "final_reward": final_reward,
            "targets_found": targets_found,
            "targets_remaining": num_targets_left,
        }
        results.append(results_i)
        env.close()

pd.DataFrame(results).to_csv(f"results_{section}.csv", sep=";", index=False)
