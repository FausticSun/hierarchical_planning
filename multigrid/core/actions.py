import enum



class Action(enum.IntEnum):
    """
    Enumeration of possible actions.
    """
    left = 0 #: Turn left
    right = enum.auto() #: Turn right
    forward = enum.auto() #: Move forward
    pickup = enum.auto() #: Pick up an object
    drop = enum.auto() #: Drop an object
    toggle = enum.auto() #: Toggle / activate an object
    done = enum.auto() #: Done completing task

class ActionUpDown(enum.IntEnum):
    """
    Enumeration of possible actions.
    """
    left = 0 #: move left
    right = enum.auto() #: move right
    up = enum.auto() #: move up 
    down = enum.auto() # move down
    pickup = enum.auto() #: Pick up an object
    drop = enum.auto() #: Drop an object
    toggle = enum.auto() #: Toggle / activate an object
    done = enum.auto() #: no-op/Done completing task

