# Parallel Execution (Fan-Out/Fan-In)
> Venkata Bhattaram (c) 2026

## Contents
* Fan-out to multiple nodes
* Supersteps and merging state
* Fan-in when branches have different lengths

## Overview
LangGraph executes nodes in **supersteps**. Nodes selected for the same
superstep run concurrently, each reading the state from the start of that
step. Their updates are merged before the next step. **Fan-out** starts
multiple branches from one node or from `START`.

If two branches write the same state field in one superstep, that field
needs a [reducer](../../03-state-management/md/reducers.md). Without one,
LangGraph raises a concurrent-update error; it does not silently pick a
winner. An `operator.add` reducer can collect both branches' list entries.

**Fan-in** continues after branch results are ready. The edge shape matters:

* `add_edge(["weather", "headlines"], "summarize")` waits for **both**
  named nodes and runs `summarize` once. This works when both branches
  always run, even if one has more steps than the other.
* Two separate edges into `summarize` can make it run more than once if
  the branches finish in different supersteps. If both happen to finish
  together, it runs once, but that is not a general join guarantee.

If fan-out is conditional and a listed branch might never run, the
list-form edge would never fire. For that case, a deferred node
(`add_node("summarize", summarize, defer=True)`) can wait for the selected
branches to finish.

## Code Example
```python
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
    return {"findings": [f"weather report for {state['topic']}"]}


def fetch_news(state: ResearchState) -> dict:
    return {"raw_news": f"news headlines for {state['topic']}"}


def extract_headlines(state: ResearchState) -> dict:
    return {"findings": [state["raw_news"]]}


def summarize(state: ResearchState) -> dict:
    # Sort only for predictable display; parallel update order may vary.
    return {
        "summary": f"{len(state['findings'])} findings: {sorted(state['findings'])}",
        "summary_runs": state["summary_runs"] + 1,
    }


builder = StateGraph(ResearchState)
builder.add_node("fetch_weather", fetch_weather)
builder.add_node("fetch_news", fetch_news)
builder.add_node("extract_headlines", extract_headlines)
builder.add_node("summarize", summarize)

# Fan-out: both nodes run in the same superstep.
builder.add_edge(START, "fetch_weather")
builder.add_edge(START, "fetch_news")
builder.add_edge("fetch_news", "extract_headlines")
# One list-form edge waits for both branch endpoints, then runs once.
builder.add_edge(["fetch_weather", "extract_headlines"], "summarize")
builder.add_edge("summarize", END)
graph = builder.compile()

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
```

The supersteps are: (1) `fetch_weather` and `fetch_news` together;
(2) `extract_headlines`; (3) `summarize` once, with both findings. The
two list updates need not arrive in a fixed order, so the example sorts
them only when making the display string. `summary_runs` makes the
once-only join visible and checks it with an assertion.

Run [the complete script](../code/parallel_execution.py) from the repo root:

```bash
python 04-control-flow/code/parallel_execution.py
```

## Conclusion
Reducers decide how parallel state updates combine; **edge shape** decides
when a join runs. Use a list-form edge to wait for a known set of branches,
and do not assume separate incoming edges always wait for each other.
Next, [Map-Reduce with Send](./map_reduce_with_send.md) creates a dynamic
number of branches at runtime.
