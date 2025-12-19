from __future__ import annotations

import random
import streamlit as st
import pandas as pd

from dataset_loader import DatasetConfig, load_gpt4all_sample
from prompt_improver import ImproveOptions, build_improvement_instructions
from llm_clients import LLMConfig, chat_completion
from scoring import analyze_prompt, compare_outputs


st.set_page_config(page_title="Prompt Improver (GPT4All dataset)", layout="wide")

st.title("Prompt Improver — Professional Prompt Upgrade + Before/After Comparison")
st.caption("Dataset: nomic-ai/gpt4all_prompt_generations_with_p3 (sampled). Works locally on Mac + deployable on streamlit.app")


@st.cache_data(show_spinner=True, ttl=60 * 60)
def cached_load(sample_size: int) -> pd.DataFrame:
    cfg = DatasetConfig(sample_size=sample_size)
    return load_gpt4all_sample(cfg)


def pick_row(df: pd.DataFrame, idx: int | None = None) -> pd.Series:
    if idx is None:
        idx = random.randint(0, len(df) - 1)
    return df.iloc[idx]


# ---------------- Sidebar ----------------
st.sidebar.header("Dataset & Model Settings")

sample_size = st.sidebar.slider("Dataset sample size (for speed)", min_value=200, max_value=5000, value=1500, step=100)
df = cached_load(sample_size)

query = st.sidebar.text_input("Search in prompts (keyword)", "")
filtered = df
if query.strip():
    q = query.strip().lower()
    filtered = df[df["prompt"].str.lower().str.contains(q, na=False)]

st.sidebar.write(f"Rows available: **{len(filtered)}** / {len(df)}")

row_index = st.sidebar.number_input("Pick row index (in filtered set)", min_value=0, max_value=max(0, len(filtered)-1), value=0, step=1)
random_btn = st.sidebar.button("🎲 Random example")

if len(filtered) == 0:
    st.error("No rows matched your search. Try another keyword.")
    st.stop()

row = pick_row(filtered, None if random_btn else int(row_index))

provider = st.sidebar.selectbox("LLM Provider", ["groq", "openai"], index=0)

# Default models (you can change)
if provider == "groq":
    model = st.sidebar.text_input("Model (Groq)", value="llama-3.1-70b-versatile")
else:
    model = st.sidebar.text_input("Model (OpenAI)", value="gpt-4o-mini")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 200, 2000, 700, 50)


st.sidebar.divider()
st.sidebar.header("Prompt Improvement Options")
role = st.sidebar.text_input("Role", value="Senior software engineer and technical writer")
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

# ---------------- Main UI ----------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Dataset example")
    st.markdown("**Prompt (original):**")
    st.code(row["prompt"], language="text")
    st.markdown("**Dataset response (reference):**")
    st.code(row["response"], language="text")
    if row.get("source", ""):
        st.caption(f"Source: {row.get('source','')}")

with colB:
    st.subheader("Improve the prompt (Professional Upgrade)")
    report = analyze_prompt(row["prompt"])
    st.markdown(f"**Prompt quality (heuristic):** `{report.clarity_score}/100`")
    if report.notes:
        st.info("• " + "\n• ".join(report.notes))

    system_improve = build_improvement_instructions(opts)

    st.markdown("**System instruction used for improvement (editable):**")
    system_improve_edit = st.text_area("", value=system_improve, height=220)

    improve_btn = st.button("✨ Generate Improved Prompt")

    improved_prompt = st.session_state.get("improved_prompt", "")

    if improve_btn:
        try:
            improved = chat_completion(
                cfg=llm_cfg,
                system=system_improve_edit,
                user=row["prompt"],
            ).strip()
            st.session_state["improved_prompt"] = improved
            improved_prompt = improved
        except Exception as e:
            st.error(str(e))

    st.markdown("**Improved prompt:**")
    improved_prompt = st.text_area(" ", value=improved_prompt, height=260)

st.divider()

st.subheader("Before / After comparison (same model, same question)")
st.caption("We run the model with: (1) original prompt, (2) improved prompt, and compare outputs vs dataset reference.")

run_compare = st.button("▶️ Run comparison")

if run_compare:
    if not improved_prompt.strip():
        st.warning("Generate an improved prompt first (or paste one).")
    else:
        try:
            with st.spinner("Running original prompt..."):
                out_original = chat_completion(
                    cfg=llm_cfg,
                    system="You are a helpful assistant.",
                    user=row["prompt"],
                ).strip()

            with st.spinner("Running improved prompt..."):
                out_improved = chat_completion(
                    cfg=llm_cfg,
                    system="You are a helpful assistant.",
                    user=improved_prompt,
                ).strip()

            st.session_state["out_original"] = out_original
            st.session_state["out_improved"] = out_improved

        except Exception as e:
            st.error(str(e))

out_original = st.session_state.get("out_original", "")
out_improved = st.session_state.get("out_improved", "")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.markdown("### Output: Original prompt")
    st.code(out_original, language="text")

with c2:
    st.markdown("### Output: Improved prompt")
    st.code(out_improved, language="text")

with c3:
    st.markdown("### Quick metrics")
    ref = row["response"]
    sim_orig = compare_outputs(ref, out_original)
    sim_impr = compare_outputs(ref, out_improved)

    st.metric("Similarity to dataset ref (orig)", f"{sim_orig}/100")
    st.metric("Similarity to dataset ref (improved)", f"{sim_impr}/100")

    # simple verdict
    if out_improved and sim_impr >= sim_orig + 5:
        st.success("Improved prompt looks closer to reference (heuristic).")
    elif out_improved and sim_impr < sim_orig - 5:
        st.warning("Improved prompt looks further from reference (heuristic).")
    elif out_improved:
        st.info("No clear change by similarity metric (heuristic).")

st.divider()
st.subheader("Tips")
st.markdown(
    """
- Sur Streamlit Cloud, **évite** de charger tout le dataset : utilise le slider sample (200–5000).
- La **similarité** n’est pas la vérité : c’est juste un indicateur “proche du style / contenu”.
- Pour une démo pro : compare un prompt brut vs un prompt avec **Role + Output Format + Constraints**.
"""
)
