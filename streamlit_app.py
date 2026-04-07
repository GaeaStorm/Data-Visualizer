import re
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from auto_viz import run_analysis


st.set_page_config(page_title="Data Inspector", layout="wide")
st.title("Data Inspector")
st.write("Upload a dataset and generate an automatic visualization of the dataset report.")


# ---------- Session state ----------
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None

if "result" not in st.session_state:
    st.session_state.result = None

if "md_report_path" not in st.session_state:
    st.session_state.md_report_path = None

if "json_report_path" not in st.session_state:
    st.session_state.json_report_path = None

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


def cleanup_previous_run():
    old_run_dir = st.session_state.get("run_dir")
    if old_run_dir and Path(old_run_dir).exists():
        shutil.rmtree(old_run_dir, ignore_errors=True)


def render_markdown_with_images(md_path: Path, output_dir: Path) -> None:
    """
    Render the markdown report inline, preserving image placement.
    When a markdown image line is encountered, render the image there.
    """
    md_text = md_path.read_text(encoding="utf-8")
    lines = md_text.splitlines()

    text_buffer = []

    def flush_text():
        nonlocal text_buffer
        if text_buffer:
            block = "\n".join(text_buffer).strip()
            if block:
                st.markdown(block)
            text_buffer = []

    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

    for line in lines:
        match = image_pattern.match(line.strip())
        if match:
            flush_text()

            alt_text = match.group(1).strip() or "Chart"
            rel_path = match.group(2).strip()
            img_path = output_dir / rel_path

            if img_path.exists():
                st.image(str(img_path), caption=alt_text, use_container_width=True)
            else:
                st.warning(f"Image not found: {rel_path}")
        else:
            text_buffer.append(line)

    flush_text()


uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "tsv", "xlsx", "xls", "parquet"],
)

max_charts = st.slider(
    "Maximum number of charts",
    min_value=5,
    max_value=60,
    value=15,
    step=5,
)

if uploaded_file is not None:
    st.write(f"Uploaded file: `{uploaded_file.name}`")

    if st.button("Run analysis", type="primary"):
        cleanup_previous_run()

        run_dir = Path(tempfile.mkdtemp(prefix="autoviz_run_"))
        input_path = run_dir / uploaded_file.name
        output_dir = run_dir / "output"

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Analyzing dataset..."):
            result = run_analysis(
                input_path=str(input_path),
                output_dir=str(output_dir),
                max_charts=max_charts,
            )

        md_report = output_dir / "analysis_report.md"
        json_report = output_dir / "analysis_report.json"

        st.session_state.run_dir = str(run_dir)
        st.session_state.result = result
        st.session_state.md_report_path = str(md_report)
        st.session_state.json_report_path = str(json_report)
        st.session_state.output_dir = str(output_dir)

        st.success(
            f"Done. Generated {result['n_charts']} charts for "
            f"{result['n_rows']} rows × {result['n_cols']} columns."
        )

# ---------- Render persisted results ----------
if st.session_state.result is not None:
    md_report = Path(st.session_state.md_report_path)
    json_report = Path(st.session_state.json_report_path)
    output_dir = Path(st.session_state.output_dir)

    if md_report.exists():
        st.subheader("Report")
        render_markdown_with_images(md_report, output_dir)
    else:
        st.error("Markdown report was not generated.")

    st.subheader("Downloads")
    col1, col2 = st.columns(2)

    with col1:
        if md_report.exists():
            st.download_button(
                "Download markdown report",
                data=md_report.read_bytes(),
                file_name="analysis_report.md",
                mime="text/markdown",
            )

    with col2:
        if json_report.exists():
            st.download_button(
                "Download JSON report",
                data=json_report.read_bytes(),
                file_name="analysis_report.json",
                mime="application/json",
            )