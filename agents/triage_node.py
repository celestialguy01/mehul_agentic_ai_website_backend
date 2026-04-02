from .state import AgentState, IntentOutput
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

TRIAGE_INSTRUCTIONS = """
    You are a strict intent classifier for a museum AI assistant.

    Classify the user input into exactly one of the following intents:

    - faq → Questions about museum information (timings, tickets, exhibitions, location, rules, etc.)
    - explain → User is asking how the AI generated its previous answer (e.g., "how did you answer", "why this answer")
    - contact → User wants to get in touch, be contacted, or needs human assistance
    - irrelevant → Input is unrelated to the museum or unclear

    Rules:
    - Do not guess missing context
    - If unsure between faq and irrelevant → choose irrelevant
    - If user asks about AI behavior → choose explain
    - If user expresses dissatisfaction or asks for help → choose contact

    Return only the intent.
    """

def triage_node(state: AgentState):
    completion = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": TRIAGE_INSTRUCTIONS},
            {"role": "user", "content": state["user_input"]},
        ],
        text_format=IntentOutput,
    )

    state["intent"] = completion.output_parsed.intent
    return state
