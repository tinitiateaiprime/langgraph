"""Compare two thread IDs using the SQLite checkpointer interface.

The lesson shows how the same graph can use PostgresSaver in production.
This local demo needs neither a PostgreSQL server nor an API key.
"""

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


def build_graph(checkpointer):
    builder = StateGraph(VisitState)
    builder.add_node("record_visit", record_visit)
    builder.add_edge(START, "record_visit")
    builder.add_edge("record_visit", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    # TemporaryDirectory keeps this comparison from changing your course data.
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "visits.sqlite3"
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(checkpointer)
            alice = {"configurable": {"thread_id": "alice"}}
            bob = {"configurable": {"thread_id": "bob"}}

            graph.invoke({"visitor": "Alice"}, alice)
            alice_again = graph.invoke({"visitor": "Alice"}, alice)
            bob_once = graph.invoke({"visitor": "Bob"}, bob)

            print("Alice visits:", alice_again["visits"])  # 2
            print("Bob visits:", bob_once["visits"])       # 1


if __name__ == "__main__":
    main()
