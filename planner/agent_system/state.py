from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
import operator
from langchain_core.messages import BaseMessage
from .utils import merge_dicts

class AnalyzedObjective(TypedDict):
    action_verb: str
    bloom_level: int
    topic: str
    grade_level: str
    constraints: List[str]

class PedagogyStrategy(TypedDict):
    chosen_pedagogy: str
    pedagogy_rationale: str
    suggested_structure: List[Dict[str, Any]]

class TeacherState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    request_type: Optional[Literal["lesson_plan", "roadmap", "clarification_needed"]]
    analyzed_objective: Optional[AnalyzedObjective]
    pedagogy_strategy: Optional[PedagogyStrategy]
    agent_outputs: Annotated[Dict[str, Any], merge_dicts]
    final_lesson_plan: Optional[str]