def parse_response(env, observations, rewards, agents,terminations, truncations, infos, num_agents):
    messages = ""

    messages += "After Step "+ str(env.unwrapped.step_count) +": \n"
    messages += "Mission:" + str(observations[0]['mission']) + "\n"
    messages += "Num of agent(s):" +  str(num_agents) + "\n"
    messages += "Num of remaining target(s):" + str(observations['global']['num_goals']) + "\n"
    for i in range(num_agents):
        messages += "Agent " + str(i) + "'s Location: " + str(observations[i]['location']) + "\n"
        messages += "Is Agent " + str(i) +" idling?: " + str(agents.idle(i)) + "\n"
        messages += "Agent " + str(i) + "'s reward:" +  str(rewards[i])+ "\n"
    messages += "Collective reward for current timestep:" + str(infos['cur_reward']) + "\n"
    messages += "Cumulative reward:"+ str(infos['total_reward']) + "\n"
    messages += "Terminations:" + str(env.unwrapped.is_done()) +  str(terminations) + "\n"
    messages += "Truncations:" + str(truncations) + "\n"
    
    return messages