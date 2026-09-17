# Checkpointers
> Venkata Bhattaram (c) 2026

## Contents
* What a checkpointer saves
* Threads and `thread_id`
* Inspecting checkpoints and their IDs

## Overview
Until now, each `graph.invoke(...)` has started with only the state you
passed to that call. A **checkpointer** records a graph's state as it runs,
so a later call can continue the same conversation or job. This is
**short-term, thread-scoped memory**: two different `thread_id` values
have separate histories.

Compile the graph with `checkpointer=...`, then pass a `thread_id` under
`configurable` on every invocation. LangGraph saves checkpoints at step
boundaries. `graph.get_state(config)` returns the latest `StateSnapshot`;
`graph.get_state_history(config)` lists that thread's snapshots, newest
first. A snapshot contains its state values, the next scheduled node(s),
and a config with its unique `checkpoint_id`.

`InMemorySaver` is useful for tests, but loses everything when Python
exits. This course begins with `SqliteSaver`: a local SQLite file that
survives process restarts. It needs the separately installed
`langgraph-checkpoint-sqlite` package from the root
[requirements.txt](../../requirements.txt), but no database server or LLM key.

## Code Example
The counter below reads the count already saved for its thread, adds one,
and saves the new value. Running the same script twice with the same
database and thread ID prints `1` and then `2`.

```python
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict, total=False):
    count: int
    step: int


def increment(state: CounterState) -> dict:
    return {"count": state.get("count", 0) + state["step"]}


builder = StateGraph(CounterState)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

db_path = Path("checkpoints.sqlite3")
with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "counter-demo"}}

    result = graph.invoke({"step": 1}, config)
    latest = graph.get_state(config)
    history = list(graph.get_state_history(config))

    print("count:", result["count"])
    print("checkpoint_id:", latest.config["configurable"]["checkpoint_id"])
    print("checkpoints in this thread:", len(history))
```

Here the input updates only `step`. The saved `count` remains in the
thread's state, so the node can read it on the next run. The history count
includes more than one snapshot per invocation; do not assume it equals
the number of times `increment` ran.

Run the [complete script](../code/checkpointers.py) from the repo root:

```bash
python 05-persistence-and-memory/code/checkpointers.py
python 05-persistence-and-memory/code/checkpointers.py
```

The script writes to `05-persistence-and-memory/data/checkpoints.sqlite3`
by default. Use `--db <path>` to choose a different file or
`--thread-id <id>` for an independent counter. Keep this database private
if it contains real conversation data.

## Conclusion
A checkpointer saves **graph state per thread**. `thread_id` chooses which
history to continue; `checkpoint_id` identifies one particular saved
snapshot. Next, [SQLite and Postgres Checkpointers](./sqlite_and_postgres_checkpointers.md)
compares local and production backends.

Official reference: [LangGraph checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers).
