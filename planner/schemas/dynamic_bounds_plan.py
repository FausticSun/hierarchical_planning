from typing import Annotated, Union, List, Dict, Literal
from pydantic import BaseModel, Field, field_serializer
from typing_extensions import TypeAlias

# Define dynamic coordinate types using Annotated
IntCoord: TypeAlias = Annotated[int, Field(ge=0)]  # Default min=0, no max yet
IntCoordBounded: TypeAlias = Annotated[int, Field(ge=0, le=100)]  # Example placeholder

# We'll create a factory function to generate the model with dynamic bounds
# But the model structure itself remains fixed and uses Annotated for runtime binding


class HLA_Move(BaseModel):
    action: Literal["move"]
    cur_x: IntCoord
    cur_y: IntCoord
    tar_x: IntCoord
    tar_y: IntCoord

    @field_serializer("action", "cur_x", "cur_y", "tar_x", "tar_y")
    def serialize_to_string(self, value, info):
        # This serializer is applied to all fields, but we only use it to reconstruct the full string
        pass  # We'll handle serialization via model_dump_json + custom __str__ or custom json_serializer

    def __str__(self):
        return f"move({self.cur_x}, {self.cur_y}, {self.tar_x}, {self.tar_y})"

    @classmethod
    def model_config(cls):
        return {"json_schema_extra": {"example": "move(1, 1, 3, 3)"}}

    # Custom JSON serializer for the entire model
    @field_serializer("action", "cur_x", "cur_y", "tar_x", "tar_y", when_used="json")
    def serialize_hla_move(self, v, info):
        # We don't serialize individual fields — we serialize the whole object as a string
        # So we override the entire model serialization via __str__ in the parent container
        # But Pydantic doesn't allow overriding model serialization directly here, so we do it in AgentHLAList
        pass


class HLA_Search(BaseModel):
    action: Literal["search"]
    cur_x: IntCoord
    cur_y: IntCoord
    x1: IntCoord
    y1: IntCoord
    x2: IntCoord
    y2: IntCoord

    def __str__(self):
        return f"search({self.cur_x}, {self.cur_y}, {self.x1}, {self.y1}, {self.x2}, {self.y2})"

    @classmethod
    def model_config(cls):
        return {"json_schema_extra": {"example": "search(1, 1, 0, 0, 5, 5)"}}

    @field_serializer(
        "action", "cur_x", "cur_y", "x1", "y1", "x2", "y2", when_used="json"
    )
    def serialize_hla_search(self, v, info):
        pass


class HLA_Stop(BaseModel):
    action: Literal["stop"]

    def __str__(self):
        return "stop()"

    @classmethod
    def model_config(cls):
        return {"json_schema_extra": {"example": "stop()"}}

    @field_serializer("action", when_used="json")
    def serialize_hla_stop(self, v, info):
        pass


# Union of all possible HLAs
HLA: TypeAlias = Union[HLA_Move, HLA_Search, HLA_Stop]


class AgentHLA(BaseModel):
    agent_id: int
    actions: List[HLA]

    @field_serializer("actions")
    def serialize_actions(self, actions: List[HLA]) -> List[str]:
        return [str(action) for action in actions]


class AgentHLAList(BaseModel):
    agents: Dict[int, List[HLA]]  # agent_id -> list of HLAs

    @field_serializer("agents")
    def serialize_agents(self, agents: Dict[int, List[HLA]]) -> Dict[int, List[str]]:
        return {
            agent_id: [str(action) for action in actions]
            for agent_id, actions in agents.items()
        }

    @classmethod
    def create_with_bounds(cls, min_coord: int, max_coord: int) -> type["AgentHLAList"]:
        """
        Factory method to dynamically create a new AgentHLAList class with updated coordinate bounds.
        Returns a new class with updated Annotated types.
        """
        global IntCoord, IntCoordBounded

        # Redefine the coordinate type with dynamic bounds
        IntCoordBounded = Annotated[int, Field(ge=min_coord, le=max_coord)]

        # Rebuild the model with the new type
        # We need to redefine the HLA types with the new IntCoordBounded
        class HLA_Move(BaseModel):
            action: Literal["move"]
            cur_x: IntCoordBounded
            cur_y: IntCoordBounded
            tar_x: IntCoordBounded
            tar_y: IntCoordBounded

            def __str__(self):
                return f"move({self.cur_x}, {self.cur_y}, {self.tar_x}, {self.tar_y})"

        class HLA_Search(BaseModel):
            action: Literal["search"]
            cur_x: IntCoordBounded
            cur_y: IntCoordBounded
            x1: IntCoordBounded
            y1: IntCoordBounded
            x2: IntCoordBounded
            y2: IntCoordBounded

            def __str__(self):
                return f"search({self.cur_x}, {self.cur_y}, {self.x1}, {self.y1}, {self.x2}, {self.y2})"

        class HLA_Stop(BaseModel):
            action: Literal["stop"]

            def __str__(self):
                return "stop()"

        HLA: TypeAlias = Union[HLA_Move, HLA_Search, HLA_Stop]

        class AgentHLA(BaseModel):
            agent_id: int
            actions: List[HLA]

            @field_serializer("actions")
            def serialize_actions(self, actions: List[HLA]) -> List[str]:
                return [str(action) for action in actions]

        class DynamicAgentHLAList(BaseModel):
            agents: Dict[int, List[HLA]]

            @field_serializer("agents")
            def serialize_agents(
                self, agents: Dict[int, List[HLA]]
            ) -> Dict[int, List[str]]:
                return {
                    agent_id: [str(action) for action in actions]
                    for agent_id, actions in agents.items()
                }

        return DynamicAgentHLAList
