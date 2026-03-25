from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_dataset_preview(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset_dir / "manifest.json"
    dataset_path = dataset_dir / "dataset.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rows: list[dict[str, Any]] = []
    if dataset_path.exists():
        for line in dataset_path.read_text(encoding="utf-8").splitlines()[:100]:
            if line.strip():
                rows.append(json.loads(line))
    return manifest, rows


def render_dashboard(dataset_dir: Path | None = None) -> None:
    import streamlit as st

    dataset_dir = dataset_dir or Path("output/test_dataset")
    st.set_page_config(page_title="TMF921 Dataset Browser", layout="wide")
    st.title("TMF921 NL Intent Dataset Browser")
    st.caption("Browse generated NL intents, TMF921 payloads, and quality metadata.")

    manifest, rows = load_dataset_preview(dataset_dir)
    st.subheader("Manifest")
    st.json(manifest)

    if not rows:
        st.warning(f"No dataset rows found in {dataset_dir}")
        return

    categories = sorted({row["metadata"]["taxonomy_category"] for row in rows})
    selected_category = st.selectbox("Taxonomy category", ["All", *categories])
    filtered_rows = rows if selected_category == "All" else [row for row in rows if row["metadata"]["taxonomy_category"] == selected_category]

    selected_index = st.slider("Row", min_value=0, max_value=len(filtered_rows) - 1, value=0)
    row = filtered_rows[selected_index]
    left, right = st.columns(2)
    with left:
        st.subheader("Natural Language Intent")
        st.write(row["nl_intent"])
        st.subheader("Metadata")
        st.json(row["metadata"])
    with right:
        st.subheader("TMF921 Intent_FVO")
        st.json(row["tmf921_intent"])


def launch_dashboard() -> None:
    script_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)], check=False)


if __name__ == "__main__":
    render_dashboard()
