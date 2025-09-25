from enum import Enum
from typing import Dict, List, Union

from pydantic import BaseModel, Field, model_serializer, model_validator


class ActionType(str, Enum):
    MOVE = "move"
    SEARCH = "search"
    STOP = "stop"


class Action(BaseModel):
    action: ActionType

    @model_serializer
    def serialize(self):
        # This will be overridden in subclasses
        raise NotImplementedError("Subclasses must implement serialize")


class StopAction(Action):
    action: ActionType = ActionType.STOP

    @model_serializer
    def serialize(self) -> str:
        return "stop()"


class MoveAction(Action):
    action: ActionType = ActionType.MOVE
    cur_x: int = Field(..., ge=0)
    cur_y: int = Field(..., ge=0)
    tar_x: int = Field(..., ge=0)
    tar_y: int = Field(..., ge=0)

    @model_serializer
    def serialize(self) -> str:
        return f"move({self.cur_x}, {self.cur_y}, {self.tar_x}, {self.tar_y})"


class SearchAction(Action):
    action: ActionType = ActionType.SEARCH
    cur_x: int = Field(..., ge=0)
    cur_y: int = Field(..., ge=0)
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_search_region(self) -> "SearchAction":
        if self.x1 > self.x2:
            raise ValueError("x1 must be <= x2")
        if self.y1 > self.y2:
            raise ValueError("y1 must be <= y2")
        return self

    @model_serializer
    def serialize(self) -> str:
        return f"search({self.cur_x}, {self.cur_y}, {self.x1}, {self.y1}, {self.x2}, {self.y2})"


# Union of all actions with discriminant field
ActionModel = Union[MoveAction, SearchAction, StopAction]


class Plan(BaseModel):
    agents: Dict[int, List[ActionModel]]
