"""Share one user's preference across two threads with a LangGraph store."""

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


def build_graph(checkpointer: SqliteSaver, store: InMemoryStore):
    builder = StateGraph(PreferenceState, context_schema=Context)
    builder.add_node("handle_request", handle_request)
    builder.add_edge(START, "handle_request")
    builder.add_edge("handle_request", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def main() -> None:
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "threads.sqlite3"
        store = InMemoryStore()
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(checkpointer, store)
            thread_1 = {"configurable": {"thread_id": "alice-1"}}
            thread_2 = {"configurable": {"thread_id": "alice-2"}}
            thread_3 = {"configurable": {"thread_id": "bob-1"}}

            saved = graph.invoke(
                {"request": "remember:blue"}, thread_1, context=Context("alice")
            )
            recalled = graph.invoke(
                {"request": "what is my favorite color?"},
                thread_2,
                context=Context("alice"),
            )
            other_user = graph.invoke(
                {"request": "what is my favorite color?"},
                thread_3,
                context=Context("bob"),
            )

            print("Alice, first thread:", saved["response"])
            print("Alice, second thread:", recalled["response"])
            print("Bob, separate namespace:", other_user["response"])


if __name__ == "__main__":
    main()
