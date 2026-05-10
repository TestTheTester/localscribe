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
    return f"""You are an expert technical note-taker.{_domain_block(domain)}
Below is a transcript from a training video. Generate comprehensive study notes.

Your output MUST follow this exact Markdown structure:

## Summary
(3-5 sentences covering the main topic and key takeaways)

## Key Concepts
(Bullet points of core ideas and definitions)

## Important Details
(Technical specifics, commands, configurations, examples, numbers)

## Action Items / Things to Remember
(Practical takeaways the viewer should do or memorise)

## Questions to Explore
(Gaps, ambiguities, or topics worth researching further)

Rules:
- Use the exact terminology from the transcript.
- Do not pad with filler or generic advice.
- If slide text is embedded as [SLIDE: ...], treat it as supplementary context.
- Flag any [uncertain] sections with a ⚠️ note.

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

Extract only the KEY POINTS from this section as concise bullet points.
Focus on: new concepts, definitions, examples, commands, and important details.
Do NOT write a final summary yet — bullet points only.

TRANSCRIPT SECTION {part}/{total}:
{transcript}
---
Respond with bullet points only. Be specific and use exact terminology."""


def build_merge_prompt(combined_points: str, domain: str = Domain.GENERAL) -> str:
    return f"""You are an expert technical note-taker.{_domain_block(domain)}
Below are key points extracted from all sections of a long training video.

Synthesise these into a complete set of notes:

## Summary
(4-6 sentences covering the full video's topic and takeaways)

## Key Concepts
(Consolidated core ideas — remove duplicates, keep all distinct concepts)

## Important Details
(Technical specifics, definitions, commands, examples)

## Action Items / Things to Remember
(Practical takeaways)

## Questions to Explore
(Gaps or topics worth researching further)

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
