from langgraph.graph import StateGraph, END
from .state import TeacherState
from .agents import objective_interpreter_agent

def build_graph():
    workflow = StateGraph(TeacherState)
    workflow.add_node("objective_interpreter", objective_interpreter_agent)
    workflow.set_entry_point("objective_interpreter")
    workflow.add_edge("objective_interpreter", END)
    return workflow.compile()

COMPILED_GRAPH = None

def get_graph_app():
    global COMPILED_GRAPH
    if COMPILED_GRAPH is None:
        COMPILED_GRAPH = build_graph()
    return COMPILED_GRAPH
