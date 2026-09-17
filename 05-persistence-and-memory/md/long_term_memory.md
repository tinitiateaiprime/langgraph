# Long-Term Memory (Store)
> Venkata Bhattaram (c) 2026

## Contents
* Checkpointer vs. store
* Namespaces and keys
* Reading and writing a store inside a graph

## Overview
A **checkpointer** saves the working state of **one thread**. A **store**
saves application-defined items that can be read by **different threads**.
For example, two conversations with the same user may need the same saved
preference even though each conversation has its own `thread_id`.

A store item has a **namespace** (a tuple, such as `("alice", "preferences")`),
a **key** (`"favorite_color"`), and a dictionary **value**
(`{"color": "blue"}`). Namespacing by user keeps Alice's preference from
appearing in Bob's conversation. `put` saves the item; `get` retrieves it.
`InMemoryStore` and database-backed stores share the `BaseStore` interface.

Compile a graph with `store=...`. A node can use the injected `Runtime`
object to access `runtime.store` and the caller's `runtime.context`. The
example below also uses a SQLite checkpointer, so each thread has its own
state while both of Alice's threads share her store namespace.

## Code Example
```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


class PreferenceState(TypedDict, total=False):
    request: str
    response: str


def handle_request(state: PreferenceState, runtime: Runtime[Context]) -> dict:
    namespace = (runtime.context.user_id, "preferences")
    request = state["request"]

    if request.startswith("remember:"):
        color = request.removeprefix("remember:").strip()
        runtime.store.put(namespace, "favorite_color", {"color": color})
        return {"response": f"Remembered {color}."}

    item = runtime.store.get(namespace, "favorite_color")
    color = item.value["color"] if item else "nothing yet"
    return {"response": f"Favorite color: {color}."}


builder = StateGraph(PreferenceState, context_schema=Context)
builder.add_node("handle_request", handle_request)
builder.add_edge(START, "handle_request")
builder.add_edge("handle_request", END)

with TemporaryDirectory() as directory:
    db_path = Path(directory) / "threads.sqlite3"
    store = InMemoryStore()
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, store=store)

        saved = graph.invoke(
            {"request": "remember:blue"},
            {"configurable": {"thread_id": "alice-1"}},
            context=Context("alice"),
        )
        recalled = graph.invoke(
            {"request": "what is my favorite color?"},
            {"configurable": {"thread_id": "alice-2"}},
            context=Context("alice"),
        )
        other_user = graph.invoke(
            {"request": "what is my favorite color?"},
            {"configurable": {"thread_id": "bob-1"}},
            context=Context("bob"),
        )

        print(saved["response"])       # Remembered blue.
        print(recalled["response"])    # Favorite color: blue.
        print(other_user["response"])  # Favorite color: nothing yet.
```

Run the [complete script](../code/long_term_memory.py):

```bash
python 05-persistence-and-memory/code/long_term_memory.py
```

The example proves **cross-thread** sharing, not durability across process
restarts: `InMemoryStore` itself lives only in RAM. For genuinely durable
long-term memory, replace it with a database-backed store such as
`PostgresStore`, set up its database schema, and keep user namespaces and
access control consistent. The SQLite checkpointer does **not** make the
separate `InMemoryStore` durable.

### Finding memories by meaning

The example knows the exact key `favorite_color`, so `get` is enough. For
many memories whose keys you do not know, `store.search(namespace)` can
list or filter items. To ask a natural-language question such as *"What
color does Alice prefer?"*, configure the store with an embedding index
(`index={"embed": your_embedding_model, "dims": embedding_size}`), save
text-bearing items, then call `store.search(namespace, query=question,
limit=3)`. That is **semantic search**: it compares meaning rather than
exact keys. It needs an embedding model (local or hosted), so this offline
beginner example does not require one; see the [official store guide](https://docs.langchain.com/oss/python/langgraph/stores)
when you are ready to add it.

## Conclusion
Use checkpointers for the state of a running thread and stores for
information shared across threads. They solve different problems and can
be compiled into the same graph. The next course section builds on this
foundation with human-in-the-loop workflows.

Official references: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence),
[stores](https://docs.langchain.com/oss/python/langgraph/stores).
