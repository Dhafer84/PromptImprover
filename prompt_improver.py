from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


@dataclass
class ImproveOptions:
    role: str = "Senior engineer"
    goal: str = "Provide an accurate, actionable answer"
    output_format: str = "Structured Markdown"
    language: str = "English"
    tone: str = "Professional"
    constraints: str = (
        "Ask clarifying questions if needed. "
        "Be concise but complete. "
        "Use steps and include assumptions."
    )


def build_improvement_instructions(opts: ImproveOptions) -> str:
    return f"""
You are a prompt engineering assistant.

Rewrite the user's prompt into a PROFESSIONAL, HIGH-QUALITY prompt.

Requirements:
- Add a clear ROLE.
- Restate the GOAL clearly.
- Add constraints (safety, correctness, scope).
- Specify an OUTPUT FORMAT.
- Keep original intent. Do NOT change the task.
- If the original prompt lacks context, add a short section called "Missing info to clarify" with 3-6 bullet questions.
- Output ONLY the improved prompt. No extra commentary.

Settings:
- Role: {opts.role}
- Goal: {opts.goal}
- Tone: {opts.tone}
- Language: {opts.language}
- Output format: {opts.output_format}
- Constraints: {opts.constraints}
""".strip()


def build_answer_prompt(system_instructions: str, improved_prompt: str, user_input: str | None = None) -> str:
    """
    Builds the final content sent to the LLM for answering using an improved prompt.
    If user_input is provided, it is appended as "INPUT:".
    """
    content = improved_prompt
    if user_input and user_input.strip():
        content += "\n\nINPUT:\n" + user_input.strip()
    return content
