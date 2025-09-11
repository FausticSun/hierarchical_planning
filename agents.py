from multigrid.core.actions import ActionUpDown

class AgentCollection:
    def __init__(self, num=0):
        self.agents = {}
        for i in range(num):
            self.add_agent(BaseAgent(i))

    def add_agent(self, agent):
        """
        Add an agent to the collection.
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name {agent.name} already exists.")
        self.agents[agent.name] = agent

    def get_agent(self, name):
        """
        Retrieve an agent by its name.
        """
        return self.agents.get(name)

    def tell(self, hla_dict):
        """
        Process a high-level action (HLA) for all agents.
        The HLA is expected to be a string that the agents can interpret.
        """
        for name, hla in hla_dict.items():
            self.agents[name].tell(hla)
            
    def act(self):
        """
        Call the `act` method on all agents and return their actions.
        """
        actions = {}
        for name, agent in self.agents.items():
            action = agent.act()
            actions[name] = action
        return actions
    
    def all_idle(self):
        """
        Check if all agents have completed their actions.
        """
        return all(agent.is_empty() for agent in self.agents.values())

    def idle(self, i):
        """
        Check if a specific agent is idle (i.e., has no actions left).
        """
        if i not in self.agents:
            raise ValueError(f"Agent with index {i} does not exist.")
        return self.agents[i].is_empty()
    
    
    def __str__(self):
        return f"AgentCollection(agents={list(self.agents.keys())})"
    
    
class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.action_queue = []

    def tell(self, hla):
        """
        Process an high-level action for the agent.
        """
        hla = hla.lower()
        if "move" in hla:
            hla = hla.split("move")[1].strip('()').strip()
            coord = hla.split(',')
            coord = [int(c.strip()) for c in coord]
            self.move(*coord)
        elif "search" in hla:
            hla = hla.split("search")[1].strip('()').strip()
            coords = hla.split(',')
            coords = [int(c.strip()) for c in coords]
            self.search(*coords)
        elif "stop" in hla:
            self.stop()
        
    def act(self):
        """
        Return the next action from the action queue.
        If the queue is empty, return no-op.
        """
        if self.action_queue:
            return self.action_queue.pop(0)
        return ActionUpDown.done  # Default action if queue is empty
    
    def is_empty(self):
        """
        Check if the action queue is empty.
        """
        return len(self.action_queue) == 0
    
    def search(self, cur_x, cur_y, x1, y1, x2, y2):
        """
        1) Move to (x1, y1).
        2) Search the area of rectangle defined by (x1, y1) and (x2, y2).
        """
        self.move(cur_x, cur_y, x1, y1)
        isRight = x1 < x2
        isDown = (y1 < y2)*2 - 1  # 1 if down, -1 if up
        w = abs(x2 - x1)
        
        for y in range(y1, y2 + 1*isDown, isDown):
            if isRight:
                self.action_queue += [ActionUpDown.right] * w
            else:
                self.action_queue += [ActionUpDown.left] * w
            if y < y2:
                self.action_queue.append(ActionUpDown.down)
            elif y > y2:
                self.action_queue.append(ActionUpDown.up)
            isRight = not isRight  # Toggle direction for next row

    def stop(self):
        """
        Stop the agent by clearing the action queue.
        """
        self.action_queue.clear()
    
    def move(self, x1, y1, x2, y2):
        """
        Move the agent from one position to another.
        """
        # move along x axis
        if x1 < x2:
            self.action_queue += [ActionUpDown.right] * (x2 - x1)
        elif x1 > x2:
            self.action_queue += [ActionUpDown.left] * (x1 - x2)
        # move along y axis
        if y1 < y2:
            self.action_queue += [ActionUpDown.down] * (y2 - y1)
        elif y1 > y2:
            self.action_queue += [ActionUpDown.up] * (y1 - y2)

    def __str__(self):
        return f"BaseAgent(name={self.name})"