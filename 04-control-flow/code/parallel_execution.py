"""Fan out to branches of different lengths, then join exactly once."""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END


class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]
    raw_news: str
    summary: str
    summary_runs: int


def fetch_weather(state: ResearchState) -> dict:
    # Runs in the same superstep as fetch_news.
    return {"findings": [f"weather report for {state['topic']}"]}


def fetch_news(state: ResearchState) -> dict:
    # This branch has an extra processing step.
    return {"raw_news": f"news headlines for {state['topic']}"}


def extract_headlines(state: ResearchState) -> dict:
    return {"findings": [state["raw_news"]]}


def summarize(state: ResearchState) -> dict:
    # The list-form edge below waits for BOTH branch endpoints. Sorting
    # gives stable presentation even if parallel updates arrive differently.
    return {
        "summary": f"{len(state['findings'])} findings: {sorted(state['findings'])}",
        "summary_runs": state["summary_runs"] + 1,
    }


builder = StateGraph(ResearchState)
builder.add_node("fetch_weather", fetch_weather)
builder.add_node("fetch_news", fetch_news)
builder.add_node("extract_headlines", extract_headlines)
builder.add_node("summarize", summarize)

# Fan-out: two edges out of START -> both nodes run in the same superstep.
builder.add_edge(START, "fetch_weather")
builder.add_edge(START, "fetch_news")
builder.add_edge("fetch_news", "extract_headlines")
# Fan-in: ONE list-form edge waits for both branch endpoints and runs
# summarize exactly once, even though the branches have different lengths.
builder.add_edge(["fetch_weather", "extract_headlines"], "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

# Step 1: fetch_weather + fetch_news; step 2: extract_headlines;
# step 3: summarize sees both findings and runs only once.
result = graph.invoke(
    {
        "topic": "San Francisco",
        "findings": [],
        "raw_news": "",
        "summary": "",
        "summary_runs": 0,
    }
)
print(result["summary"])
print("summary ran:", result["summary_runs"])  # -> 1
assert result["summary_runs"] == 1
# -> 2 findings: ['news headlines for San Francisco',
#                 'weather report for San Francisco']
