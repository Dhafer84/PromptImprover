from __future__ import annotations

import random
import streamlit as st
import pandas as pd

from dataset_loader import DatasetConfig, load_gpt4all_sample
from prompt_improver import ImproveOptions, build_improvement_instructions
from llm_clients import LLMConfig, chat_completion
from scoring import analyze_prompt


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Prompt Improver", layout="wide")

st.title("Prompt Improver — Guided Demo")
st.caption("Follow the steps: 1) Choose  2) Improve  3) Test (Before/After)")


# =========================
# Cache dataset sample
# =========================
@st.cache_data(show_spinner=True, ttl=60 * 60)
def cached_load(sample_size: int) -> pd.DataFrame:
    cfg = DatasetConfig(sample_size=sample_size)
    return load_gpt4all_sample(cfg)


def pick_row(df: pd.DataFrame, idx: int | None = None) -> pd.Series:
    if idx is None:
        idx = random.randint(0, len(df) - 1)
    return df.iloc[idx]


# =========================
# Sidebar — Dataset & Model
# =========================
st.sidebar.header("Dataset & Model Settings")

sample_size = st.sidebar.slider(
    "Dataset sample size (for speed)",
    min_value=200,
    max_value=5000,
    value=1500,
    step=100,
)

df = cached_load(sample_size)

query = st.sidebar.text_input("Search in prompts (keyword)", "")
filtered = df
if query.strip():
    q = query.strip().lower()
    filtered = df[df["prompt"].str.lower().str.contains(q, na=False)]

st.sidebar.write(f"Rows available: **{len(filtered)} / {len(df)}**")

if len(filtered) == 0:
    st.error("No rows matched your search. Try another keyword.")
    st.stop()

row_index = st.sidebar.number_input(
    "Pick row index (in filtered set)",
    min_value=0,
    max_value=max(0, len(filtered) - 1),
    value=0,
    step=1,
)

random_btn = st.sidebar.button("🎲 Random example")
row = pick_row(filtered, None if random_btn else int(row_index))

provider = st.sidebar.selectbox("LLM Provider", ["groq", "openai"], index=0)

st.sidebar.subheader("Model")
if provider == "groq":
    # Mets ici des modèles Groq valides chez toi
    model = st.sidebar.selectbox(
        "Model (Groq)",
        [
            "llama-3.1-8b-instant",
            "llama-3.2-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )
else:
    model = st.sidebar.text_input("Model (OpenAI)", value="gpt-4o-mini")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 200, 2000, 700, 50)

st.sidebar.divider()
st.sidebar.header("Prompt Improvement Options")

role = st.sidebar.text_input("Role", value="Senior software engineer")
goal = st.sidebar.text_input("Goal", value="Deliver a correct, actionable, well-structured answer")
tone = st.sidebar.selectbox("Tone", ["Professional", "Concise", "Strict", "Friendly"], index=0)
language = st.sidebar.selectbox("Language", ["English", "French"], index=0)
output_format = st.sidebar.selectbox("Output format", ["Structured Markdown", "Table (Markdown)", "JSON"], index=0)
constraints = st.sidebar.text_area(
    "Constraints",
    value="Ask clarifying questions if needed. Provide step-by-step guidance. Mention assumptions. Avoid hallucinations and be explicit about uncertainty.",
    height=110,
)

opts = ImproveOptions(
    role=role,
    goal=goal,
    tone=tone,
    language=language,
    output_format=output_format,
    constraints=constraints,
)

llm_cfg = LLMConfig(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)


# =========================
# Session state init
# =========================
if "current_prompt" not in st.session_state:
    st.session_state["current_prompt"] = ""

if "improved_prompt" not in st.session_state:
    st.session_state["improved_prompt"] = ""

if "out_original" not in st.session_state:
    st.session_state["out_original"] = ""

if "out_improved" not in st.session_state:
    st.session_state["out_improved"] = ""

if "model_orig" not in st.session_state:
    st.session_state["model_orig"] = ""

if "model_impr" not in st.session_state:
    st.session_state["model_impr"] = ""


# =========================
# STEP 1 — Choose prompt
# =========================
with st.expander("Step 1 — Choose a prompt", expanded=True):
    mode = st.radio(
        "Select input mode",
        ["Dataset example (filtered)", "Manual (paste your prompt)"],
        horizontal=True
    )

    if mode == "Dataset example (filtered)":
        st.info("Use the left sidebar filters (keyword, index, random), then load the selected prompt here.")
        if st.button("📥 Load selected dataset prompt"):
            st.session_state["current_prompt"] = row["prompt"]
            # reset previous outputs when changing prompt
            st.session_state["improved_prompt"] = ""
            st.session_state["out_original"] = ""
            st.session_state["out_improved"] = ""
            st.session_state["model_orig"] = ""
            st.session_state["model_impr"] = ""

    else:
        manual = st.text_area(
            "Paste your prompt",
            height=180,
            placeholder="Example: I need a MISRA-C compliant code review checklist for STM32 drivers..."
        )
        if st.button("📥 Use this prompt"):
            st.session_state["current_prompt"] = manual
            st.session_state["improved_prompt"] = ""
            st.session_state["out_original"] = ""
            st.session_state["out_improved"] = ""
            st.session_state["model_orig"] = ""
            st.session_state["model_impr"] = ""

    st.markdown("**Selected prompt (scrollable):**")
    st.text_area(
        label="",
        value=st.session_state["current_prompt"],
        height=160,
        disabled=True
    )
    st.caption("Tip: click inside then ⌘A → ⌘C to copy. (Mac)")


# =========================
# STEP 2 — Improve prompt
# =========================
with st.expander("Step 2 — Improve the prompt (Professional Upgrade)", expanded=True):
    current = st.session_state["current_prompt"].strip()

    if not current:
        st.warning("Load or paste a prompt in Step 1.")
    else:
        report = analyze_prompt(current)
        st.markdown(f"**Prompt quality (heuristic):** `{report.clarity_score}/100`")
        if report.notes:
            st.info("• " + "\n• ".join(report.notes))

        system_improve = build_improvement_instructions(opts)

        st.markdown("**System instruction used for improvement (editable):**")
        system_improve_edit = st.text_area(
            "",
            value=system_improve,
            height=200
        )

        if st.button("✨ Generate Improved Prompt"):
            try:
                improved, used_model = chat_completion(
                    cfg=llm_cfg,
                    system=system_improve_edit,
                    user=current
                )
                st.session_state["improved_prompt"] = (improved or "").strip()

                # Warning fallback Groq
                if provider == "groq" and used_model != llm_cfg.model:
                    st.warning(f"⚠️ Model `{llm_cfg.model}` is deprecated. Fallback to `{used_model}`.")

            except Exception as e:
                st.error(str(e))

        st.markdown("**Improved prompt (scrollable):**")
        st.text_area(
            label="",
            value=st.session_state["improved_prompt"],
            height=220
        )


# =========================
# STEP 3 — Before/After test
# =========================
with st.expander("Step 3 — Test (Before / After)", expanded=True):
    current = st.session_state["current_prompt"].strip()
    improved = st.session_state["improved_prompt"].strip()

    if not current:
        st.warning("Step 1 is required (select a prompt).")
    elif not improved:
        st.warning("Step 2 is required (generate the improved prompt).")
    else:
        if st.button("▶️ Run Before/After comparison"):
            try:
                with st.spinner("Running with original prompt..."):
                    out_original, model_orig = chat_completion(
                        cfg=llm_cfg,
                        system="You are a helpful assistant.",
                        user=current
                    )

                with st.spinner("Running with improved prompt..."):
                    out_improved, model_impr = chat_completion(
                        cfg=llm_cfg,
                        system="You are a helpful assistant.",
                        user=improved
                    )

                st.session_state["out_original"] = (out_original or "").strip()
                st.session_state["out_improved"] = (out_improved or "").strip()
                st.session_state["model_orig"] = model_orig
                st.session_state["model_impr"] = model_impr

                if provider == "groq":
                    if model_orig != llm_cfg.model:
                        st.warning(f"⚠️ Original run fallback: `{llm_cfg.model}` → `{model_orig}`.")
                    if model_impr != llm_cfg.model:
                        st.warning(f"⚠️ Improved run fallback: `{llm_cfg.model}` → `{model_impr}`.")

            except Exception as e:
                st.error(str(e))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Output — Original")
            if st.session_state.get("model_orig"):
                st.caption(f"Model used: `{st.session_state['model_orig']}`")
            st.text_area(
                "",
                value=st.session_state.get("out_original", ""),
                height=280
            )

        with col2:
            st.markdown("### Output — Improved")
            if st.session_state.get("model_impr"):
                st.caption(f"Model used: `{st.session_state['model_impr']}`")
            st.text_area(
                "",
                value=st.session_state.get("out_improved", ""),
                height=280
            )


# =========================
# Small help section
# =========================
with st.expander("ℹ️ Help / Tips", expanded=False):
    st.markdown(
        """
- **Step 1**: Choose a dataset prompt (via sidebar) or paste your own.
- **Step 2**: Generate a professional improved prompt (Role + Goal + Output format + Constraints).
- **Step 3**: Compare model answers before/after.
- If Groq shows **model_decommissioned**, the app will automatically fallback and show a warning.
"""
    )
