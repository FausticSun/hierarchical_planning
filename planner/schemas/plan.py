from enum import Enum
from typing import Dict, List, Union

from pydantic import BaseModel, Field, model_serializer, model_validator


class ActionType(str, Enum):
    MOVE = "move"


class Action(BaseModel):
    type: ActionType

    @model_serializer
    def serialize(self):
        # This will be overridden in subclasses
        raise NotImplementedError("Subclasses must implement serialize")


class MoveAction(Action):
    type: ActionType = ActionType.MOVE
    cur_x: int = Field(..., ge=0)
    cur_y: int = Field(..., ge=0)
    tar_x: int = Field(..., ge=0)
    tar_y: int = Field(..., ge=0)

    @model_serializer
    def serialize(self) -> str:
        return f"move({self.cur_x}, {self.cur_y}, {self.tar_x}, {self.tar_y})"


# Union of all actions with discriminant field
ActionModel = Union[MoveAction]


class Plan(BaseModel):
    agents: Dict[int, List[ActionModel]]
