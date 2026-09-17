"""Keep a counter across separate runs with a SQLite checkpointer.

Run this script twice with the same --db and --thread-id to see the count grow.
No LLM or API key is needed.
"""

import argparse
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict, total=False):
    count: int
    step: int


def increment(state: CounterState) -> dict:
    return {"count": state.get("count", 0) + state["step"]}


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    default_db = Path(__file__).resolve().parents[1] / "data" / "checkpoints.sqlite3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=default_db)
    parser.add_argument("--thread-id", default="counter-demo")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(args.db)) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        # The saved count is restored for this thread before the node runs.
        result = graph.invoke({"step": 1}, config)
        snapshot = graph.get_state(config)
        history = list(graph.get_state_history(config))

        print("count:", result["count"])
        print("checkpoint_id:", snapshot.config["configurable"]["checkpoint_id"])
        print("checkpoints in this thread:", len(history))
        print("database:", args.db)


if __name__ == "__main__":
    main()
