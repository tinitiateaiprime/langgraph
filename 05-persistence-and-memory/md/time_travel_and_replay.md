# Time Travel and Replay
> Venkata Bhattaram (c) 2026

## Contents
* Inspecting checkpoint history
* Replaying from an earlier checkpoint
* Forking with edited state

## Overview
Once a graph has a checkpointer, you can inspect where it has been.
`graph.get_state_history(config)` returns snapshots newest first. Each
snapshot's `values` are the saved state; `next` lists nodes that would run
when resumed; `config` includes the `checkpoint_id` needed to select that
exact point.

**Replay** calls `graph.invoke(None, saved_snapshot.config)`. It does not
merely read an old result: it **runs the nodes after that checkpoint again**.
Nodes before the checkpoint are not rerun. **Fork** first calls
`graph.update_state(saved_snapshot.config, updates)` to make a new
checkpoint with changed state, then resumes from the returned config.
Forking does not erase the original history.

## Code Example
```python
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class ArticleState(TypedDict, total=False):
    topic: str
    draft: str
    final: str


def normalize_topic(state: ArticleState) -> dict:
    return {"topic": state["topic"].strip()}


def write_draft(state: ArticleState) -> dict:
    print("write_draft ran for:", state["topic"])
    return {"draft": f"A short guide to {state['topic']}."}


def publish(state: ArticleState) -> dict:
    return {"final": state["draft"] + " Ready to publish."}


builder = StateGraph(ArticleState)
builder.add_node("normalize_topic", normalize_topic)
builder.add_node("write_draft", write_draft)
builder.add_node("publish", publish)
builder.add_edge(START, "normalize_topic")
builder.add_edge("normalize_topic", "write_draft")
builder.add_edge("write_draft", "publish")
builder.add_edge("publish", END)

with TemporaryDirectory() as directory:
    db_path = Path(directory) / "time-travel.sqlite3"
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "article-demo"}}

        original = graph.invoke({"topic": "LangGraph"}, config)
        original_final_config = graph.get_state(config).config
        history = list(graph.get_state_history(config))
        before_draft = next(
            snapshot for snapshot in history
            if snapshot.next == ("write_draft",)
        )
        print("original:", original["final"])
        print("checkpoint_id:", before_draft.config["configurable"]["checkpoint_id"])

        replayed = graph.invoke(None, before_draft.config)
        print("replayed:", replayed["final"])

        fork_config = graph.update_state(
            before_draft.config,
            {"topic": "Persistence"},
            as_node="normalize_topic",
        )
        forked = graph.invoke(None, fork_config)
        print("forked:", forked["final"])
        print("original still exists:", graph.get_state(original_final_config).values["final"])
```

We locate the checkpoint by `next == ("write_draft",)` instead of relying
on a fragile numeric position in history. The explicit `as_node` tells
LangGraph to treat the edit as coming from `normalize_topic`, so execution
continues with its successor, `write_draft`.

Run the [complete script](../code/time_travel_and_replay.py):

```bash
python 05-persistence-and-memory/code/time_travel_and_replay.py
```

**Important:** replaying a node repeats its effects too. An LLM call can
return a different answer; a payment, email, or database write could be
performed again. Use idempotency keys or separate human approval for
external actions you might replay.

## Conclusion
Checkpoint history is not just a log: it can be a starting point for a new
execution. Replay keeps the saved state and reruns later nodes; fork first
edits state and creates a new branch. Next, [Long-Term Memory](./long_term_memory.md)
stores information that should be available across different threads.

Official reference: [LangGraph time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel).
