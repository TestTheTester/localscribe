"""
llm/prompts.py — All prompt templates, keyed by domain and task.

Each build_* function injects the domain-specific context block so the LLM
knows what vocabulary and depth of detail to apply.
"""

from config import Domain

# ── Domain context injected into every prompt ──────────────────────────────────

DOMAIN_CONTEXT: dict[str, str] = {
    Domain.GENERAL: (
        "Apply general best-practice note-taking. Cover all topics mentioned."
    ),
    Domain.SECURITY: (
        "Focus on: attack techniques, CVEs, MITRE ATT&CK tactics/techniques, "
        "defensive controls, threat actors, vulnerability classes, tools (Metasploit, "
        "Burp Suite, Nmap, etc.), and security frameworks (NIST, CIS, ISO 27001)."
    ),
    Domain.DEVOPS: (
        "Focus on: CI/CD concepts, containerisation (Docker, Kubernetes), "
        "infrastructure-as-code (Terraform, Ansible, Pulumi), cloud services "
        "(AWS, Azure, GCP), SRE principles, monitoring/observability, and deployment patterns."
    ),
    Domain.PROGRAMMING: (
        "Focus on: language features, design patterns, algorithms, data structures, "
        "APIs, library/framework usage, performance considerations, and code examples "
        "or snippets mentioned."
    ),
    Domain.DATA: (
        "Focus on: ML algorithms, model architectures, training techniques, "
        "evaluation metrics, data preprocessing, libraries (PyTorch, TensorFlow, "
        "scikit-learn, Pandas), statistical concepts, and experiment methodology."
    ),
    Domain.NETWORKING: (
        "Focus on: network protocols (TCP/IP, DNS, BGP, OSPF), OSI/TCP-IP model layers, "
        "device roles (routers, switches, firewalls), packet analysis, subnetting, "
        "VLANs, VPNs, and troubleshooting methodologies."
    ),
    Domain.BUSINESS: (
        "Focus on: strategic frameworks, methodologies (Agile, Scrum, Lean, OKRs), "
        "organisational concepts, leadership principles, financial metrics, "
        "process improvement, and actionable management takeaways."
    ),
}


def _domain_block(domain: str) -> str:
    ctx = DOMAIN_CONTEXT.get(domain, DOMAIN_CONTEXT[Domain.GENERAL])
    return f"\nDOMAIN FOCUS — {domain.upper()}:\n{ctx}\n"


# ── Main note-generation prompts ───────────────────────────────────────────────

def build_summary_prompt(transcript: str, domain: str = Domain.GENERAL) -> str:
    return f"""You are an expert technical note-taker and educator.{_domain_block(domain)}
Below is a transcript from a training video. Generate thorough, detailed study notes \
that would let someone who never watched the video fully understand the material.

Your output MUST follow this exact Markdown structure:

## Summary
(8-12 sentences. Cover: what the video is about, the central problem or goal, the \
approach taken, key conclusions, and why this topic matters in the broader domain.)

## Key Concepts
For each concept: write the name as a **bold sub-heading**, then 2-4 sentences \
defining it, explaining how it works, and why it is significant. Cover every distinct \
concept, technique, or term introduced in the video.

## Important Details
Technical specifics with full context: commands and their flags, configuration values \
and what they control, step-by-step procedures, exact numbers/thresholds, error \
conditions, and worked examples mentioned. Use sub-bullets to group related details.

## Extended Concepts
(Concepts that were **mentioned or briefly referenced** in the video but not fully \
explained. For each one, use your own knowledge to provide 2-4 sentences of \
background explanation so the reader understands it without needing to look it up. \
Label each with: **[concept name]** — *not covered in depth in the video*.)

## Action Items / Things to Remember
Practical takeaways: what to do, what to avoid, what to set up, what to memorise. \
Include the reasoning behind each item, not just the item itself.

## Questions to Explore
Gaps, ambiguities, or related topics worth researching further. For each question, \
add one sentence on why it matters or what answering it would unlock.

Rules:
- Use the exact terminology from the transcript; do not paraphrase technical terms.
- Do not pad with filler or generic advice.
- If slide text is embedded as [SLIDE: ...], treat it as supplementary context.
- Flag any [uncertain] transcript sections with a ⚠️ note.
- Aim for depth over brevity — a thorough set of notes is the goal.

TRANSCRIPT:
{transcript}
---
Respond in clean Markdown. Start with ## Summary."""


def build_chunk_summary_prompt(
    transcript: str,
    part: int,
    total: int,
    domain: str = Domain.GENERAL,
) -> str:
    return f"""You are an expert technical note-taker.{_domain_block(domain)}
Below is PART {part} of {total} from a long training video transcript.

Extract detailed points from this section. Use two levels of bullets:
- Top-level bullet: the concept, step, or topic introduced.
  - Sub-bullet: explanation, example, command, value, or "why it matters" context.

Cover: new concepts and their definitions, procedures and their steps, commands and \
flags, configuration details, concrete examples, and any terms that are mentioned \
but not explained (tag these with [NEEDS EXPLANATION]).

Do NOT write a final summary yet — structured bullet points only.

TRANSCRIPT SECTION {part}/{total}:
{transcript}
---
Respond with structured bullet points only. Be specific and use exact terminology."""


def build_merge_prompt(combined_points: str, domain: str = Domain.GENERAL) -> str:
    return f"""You are an expert technical note-taker and educator.{_domain_block(domain)}
Below are detailed points extracted from all sections of a long training video. \
Synthesise them into a thorough, complete set of study notes.

Your output MUST follow this exact Markdown structure:

## Summary
(8-12 sentences. Cover: what the video is about, the central problem or goal, the \
approach taken, key conclusions, and why this topic matters in the broader domain.)

## Key Concepts
For each concept: write the name as a **bold sub-heading**, then 2-4 sentences \
defining it, explaining how it works, and why it is significant. Consolidate \
duplicates but keep every distinct concept.

## Important Details
Technical specifics with full context: commands and their flags, configuration values \
and what they control, step-by-step procedures, exact numbers/thresholds, and worked \
examples. Use sub-bullets to group related details.

## Extended Concepts
Items tagged [NEEDS EXPLANATION] in the source points, plus any other terms that were \
mentioned but not fully explained. For each one, use your own knowledge to provide \
2-4 sentences of background so the reader understands without looking it up. \
Label each: **[concept name]** — *not covered in depth in the video*.

## Action Items / Things to Remember
Practical takeaways with the reasoning behind each item.

## Questions to Explore
Gaps or related topics worth researching further. For each, add one sentence on why \
it matters.

EXTRACTED POINTS FROM ALL SECTIONS:
{combined_points}
---
Respond in clean Markdown. Start with ## Summary."""


# ── Anki flashcard generation ──────────────────────────────────────────────────

def build_anki_prompt(
    notes_text: str,
    domain: str = Domain.GENERAL,
    max_cards: int = 20,
) -> str:
    return f"""You are an expert educator creating Anki flashcards.{_domain_block(domain)}
Based on the notes below, generate up to {max_cards} high-quality flashcard Q&A pairs.

Rules:
- Each card tests ONE specific fact, concept, or definition.
- Questions should be precise and unambiguous.
- Answers should be concise (1-3 sentences or a short list).
- Prioritise the most important and testable content.
- Do NOT create trivial or overly broad cards.

Output format — one card per block, exactly like this:
Q: <question>
A: <answer>

(blank line between cards)

NOTES:
{notes_text}
---
Generate the flashcards now:"""


# ── Post-session Q&A chat ──────────────────────────────────────────────────────

CHAT_SYSTEM = """You are an intelligent study assistant. The user has just finished \
watching a training video. You have access to the full transcript and the generated \
notes. Answer questions accurately using this material. If the answer is not in the \
material, say so and offer what you do know. Be concise but complete."""


def build_chat_messages(
    transcript: str,
    notes: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """
    Construct the full messages list for a multi-turn chat.
    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    context = (
        f"=== TRAINING NOTES ===\n{notes}\n\n"
        f"=== FULL TRANSCRIPT ===\n{transcript}"
    )
    messages = [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user",   "content": f"Here is the session material:\n\n{context}"},
        {"role": "assistant", "content": "Got it. I have the transcript and notes. Ask me anything."},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
