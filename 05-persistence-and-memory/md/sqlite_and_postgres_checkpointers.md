# SQLite and Postgres Checkpointers
> Venkata Bhattaram (c) 2026

## Contents
* Choosing a backend
* Separate state for separate threads
* Switching to Postgres

## Overview
A graph uses the same `compile(checkpointer=...)`, `thread_id`,
`get_state`, and `get_state_history` interface regardless of the backend.
The backend changes **where** snapshots live and how the application can
operate:

| Backend | Good for | Important limit |
|---|---|---|
| `InMemorySaver` | Tests and disposable demos | Lost when the process exits |
| `SqliteSaver` | Local development and single-process jobs | A local file is not a shared, high-concurrency service |
| `PostgresSaver` | Deployed applications with a managed database | Needs PostgreSQL, credentials, and schema setup |

The code below uses SQLite and two thread IDs. Alice's second call
continues **her** saved state, while Bob starts from his own empty state.
It puts the file in a temporary directory so this comparison leaves no
course data behind. For a database that survives script restarts, use the
previous lesson's persistent file.

## Code Example
```python
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class VisitState(TypedDict, total=False):
    visits: int
    visitor: str


def record_visit(state: VisitState) -> dict:
    return {"visits": state.get("visits", 0) + 1}


builder = StateGraph(VisitState)
builder.add_node("record_visit", record_visit)
builder.add_edge(START, "record_visit")
builder.add_edge("record_visit", END)

with TemporaryDirectory() as directory:
    db_path = Path(directory) / "visits.sqlite3"
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        alice = {"configurable": {"thread_id": "alice"}}
        bob = {"configurable": {"thread_id": "bob"}}

        graph.invoke({"visitor": "Alice"}, alice)
        alice_again = graph.invoke({"visitor": "Alice"}, alice)
        bob_once = graph.invoke({"visitor": "Bob"}, bob)

        print("Alice visits:", alice_again["visits"])  # 2
        print("Bob visits:", bob_once["visits"])       # 1
```

Run the [complete script](../code/sqlite_and_postgres_checkpointers.py):

```bash
python 05-persistence-and-memory/code/sqlite_and_postgres_checkpointers.py
```

### Switching to Postgres

For a deployed service, install the optional Postgres packages and supply
a connection URI through an environment variable. Do **not** paste a real
password into lesson code or commit it to the repository.

```bash
pip install "langgraph-checkpoint-postgres>=3.0.0" "psycopg[binary,pool]>=3.2.0"
```

Replace the SQLite `with` block above with this block after defining the
same `builder`. Run `setup()` when preparing a new database (prefer a
deployment migration step in production):

```python
import os
from langgraph.checkpoint.postgres import PostgresSaver

db_uri = os.environ["LANGGRAPH_POSTGRES_URI"]
with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    checkpointer.setup()  # create/update checkpointer tables when needed
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "alice"}}
    print(graph.invoke({"visitor": "Alice"}, config)["visits"])
```

For an async graph, use the corresponding `AsyncSqliteSaver` or
`AsyncPostgresSaver` with `async with`, `await graph.ainvoke(...)`, and
`await checkpointer.setup()` for the Postgres saver. Keep the saver open
for as long as the compiled graph uses it.

## Conclusion
The graph logic and `thread_id` contract stay the same while the storage
backend changes. SQLite is an easy local starting point; Postgres is a
better fit when multiple application instances must share durable state.
Next, [Time Travel and Replay](./time_travel_and_replay.md) uses individual
checkpoints to revisit an earlier step.

Official references: [checkpointer implementations](https://docs.langchain.com/oss/python/langgraph/checkpointers),
[memory setup](https://docs.langchain.com/oss/python/langgraph/add-memory).
