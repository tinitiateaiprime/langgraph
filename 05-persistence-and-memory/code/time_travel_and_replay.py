"""Inspect a checkpoint, replay from it, then fork with different state."""

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


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(ArticleState)
    builder.add_node("normalize_topic", normalize_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_node("publish", publish)
    builder.add_edge(START, "normalize_topic")
    builder.add_edge("normalize_topic", "write_draft")
    builder.add_edge("write_draft", "publish")
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=checkpointer)


def main() -> None:
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "time-travel.sqlite3"
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(checkpointer)
            config = {"configurable": {"thread_id": "article-demo"}}

            original = graph.invoke({"topic": "LangGraph"}, config)
            original_final_config = graph.get_state(config).config
            history = list(graph.get_state_history(config))  # newest first
            before_draft = next(
                snapshot for snapshot in history
                if snapshot.next == ("write_draft",)
            )
            print("original:", original["final"])
            print("saved topic:", before_draft.values["topic"])
            print("checkpoint_id:", before_draft.config["configurable"]["checkpoint_id"])

            # Replays write_draft and publish; normalize_topic is not rerun.
            replayed = graph.invoke(None, before_draft.config)
            print("replayed:", replayed["final"])

            # update_state creates a NEW branch; it does not erase the original.
            fork_config = graph.update_state(
                before_draft.config,
                {"topic": "Persistence"},
                as_node="normalize_topic",
            )
            forked = graph.invoke(None, fork_config)
            print("forked:", forked["final"])
            print("original still exists:", graph.get_state(original_final_config).values["final"])


if __name__ == "__main__":
    main()
