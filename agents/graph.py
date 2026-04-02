from .state import AgentState
from .faq_node import faq_node
#from explain_node import explain_node
#from contact_node import contact_node
#from fallback_node import fallback_node
from .triage_node import triage_node
from langgraph.graph import StateGraph


def explain_node(state: AgentState):
    state["response"] = (
        "The FAQ assistant follows a two-stage retrieval process. "
        "Each query is first evaluated against a curated FAQ dataset to identify high-confidence matches. "
        "If no reliable match is found, the system transitions to a broader knowledge retrieval step to construct a grounded response. "
        "This ensures precision for common queries while maintaining coverage for less structured questions."
    )
    return state

def contact_node(state: AgentState):
    state["response"] = (
        "For more specific queries or assistance beyond the available information, "
        "you can fill the contact form on the website and I’ll help connect you accordingly.\n\n"
        "Or reach out directly:\n"
        "📧 itsmehul01@gmail.com\n"
        "📞 +91 9623394530"
    )
    return state

def fallback_node(state: AgentState):
    state["response"] = (
        "I couldn’t find a clear answer to that within the available information. "
        "You can try rephrasing your question, or I can help you get in touch for more specific assistance."
    )
    return state

def route_by_intent(state: AgentState):
    intent = state["intent"]

    if intent == "faq":
        return "faq_node"

    elif intent == "explain":
        return "explain_node"

    elif intent == "contact":
        return "contact_node"

    elif intent == "irrelevant":
        return "fallback_node"

    else:
        return "fallback_node"

builder = StateGraph(AgentState)

builder.add_node("triage_node", triage_node)

builder.add_node("faq_node", faq_node)
builder.add_node("explain_node", explain_node)
builder.add_node("contact_node", contact_node)
builder.add_node("fallback_node", fallback_node)

builder.add_conditional_edges(
    "triage_node",
    route_by_intent,
    {
        "faq_node": "faq_node",
        "explain_node": "explain_node",
        "contact_node": "contact_node",
        "fallback_node": "fallback_node",
    },
)

builder.set_entry_point("triage_node")

graph = builder.compile()