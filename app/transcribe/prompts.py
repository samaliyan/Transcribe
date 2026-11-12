"""Prompts sent to the LLM, and the ready-made prompts offered in the settings."""

from . import constants

# Shown in the reply panel until the first suggestion arrives.
INITIAL_RESPONSE = ''

ME_LABEL = 'Me'
OTHER_LABEL = 'Other'

# name -> system prompt. The first one is the default (also in parameters.yaml).
PROMPT_PRESETS = {
    'Meeting / call assistant': (
        "You are my real-time assistant during a live conversation (a meeting, a call or an interview). "
        "You receive a rough automatic transcript: lines marked 'Me' are what I said, lines marked "
        "'Other' are what the other people said. It can contain recognition mistakes; work out what was meant. "
        "Give me what I should say next, or the answer to the latest question from the others. "
        "Be correct and specific, and short enough to read at a glance: a few sentences or short bullet points. "
        "Never ask me to repeat or clarify, and do not describe or summarize the transcript."
    ),
    'Job interview (I am the candidate)': (
        "I am a candidate in a live job interview and you help me answer. You receive a rough automatic "
        "transcript: lines marked 'Other' are the interviewer, lines marked 'Me' are my own words. It can "
        "contain recognition mistakes; work out what was meant. Answer the interviewer's latest question the "
        "way a strong candidate would say it out loud: first the direct answer, then the key points or a "
        "short example. Technical questions get a precise, correct answer. Keep it short enough to read at a "
        "glance. Never ask me to repeat or clarify."
    ),
    'Explain what is being said': (
        "You help me follow a live conversation. You receive a rough automatic transcript: lines marked "
        "'Other' are other people, lines marked 'Me' are my own words. Explain in simple words what the "
        "others just said or asked, including any technical terms, then suggest one short reply I could give."
    ),
    'Casual conversation': (
        "You are a friendly companion in a casual conversation. You receive a rough automatic transcript: "
        "lines marked 'Me' are what I said, lines marked 'Other' are what the other people said. Suggest a "
        "natural, relaxed reply I could give next. Never ask me to repeat or clarify."
    ),
}
DEFAULT_SYSTEM_PROMPT = next(iter(PROMPT_PRESETS.values()))


def with_response_language(prompt: str, response_lang) -> str:
    """Add the reply language to a system prompt."""
    prompt = (prompt or '').strip()
    if response_lang:
        prompt = f'{prompt.rstrip(".")}. Respond exclusively in {str(response_lang).title()}.'
    return prompt


def strip_brackets(text: str) -> str:
    """'[text]' -> 'text'. Older prompts asked the model to answer in square brackets."""
    text = (text or '').strip()
    if len(text) >= 2 and text[0] == '[' and text[-1] == ']' and '[' not in text[1:-1]:
        return text[1:-1].strip()
    if text.startswith('[') and ']' not in text:      # still streaming: the closing bracket comes last
        return text[1:].lstrip()
    return text


def split_row(row_text: str):
    """'You: [hello]\\n\\n' -> ('You', 'hello'). Rows are stored this way by the conversation."""
    persona, separator, content = row_text.partition(':')
    if not separator:
        return '', row_text.strip()
    content = content.strip()
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]
    return persona.strip(), content.strip()


def create_prompt_for_text(text: str, config: dict):
    """Messages asking for a reply to the text the user selected in the transcript."""
    if text is None or text.strip() == '':
        return None
    general = config["General"]
    system_prompt = with_response_language(general["system_prompt"], config["OpenAI"]["response_lang"])
    preamble = (general.get("default_prompt_preamble") or '').strip()
    epilogue = (general.get("default_prompt_epilogue") or '').strip()
    user_text = '\n\n'.join(part for part in (preamble, text.strip(), epilogue) if part)
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}]


def create_multiturn_prompt(convo: list[tuple]) -> list[dict[str, str]]:
    """Turn conversation rows into chat messages.

    Each row is a tuple whose first item is 'Persona: [text]'. The microphone ('You') and the
    speakers ('Speaker') are both the user's side of the chat, so their consecutive rows become
    one user message in which every line says who spoke ('Me: ...' / 'Other: ...'); without the
    labels the model could not tell a question it should answer from the user's own words.
    """
    labels = {constants.PERSONA_YOU.lower(): ME_LABEL, constants.PERSONA_SPEAKER.lower(): OTHER_LABEL}
    messages: list[dict[str, str]] = []
    for convo_item in convo:
        persona, content = split_row(convo_item[0])
        if not content:
            continue
        label = labels.get(persona.lower())
        if label is None:               # system or assistant
            messages.append({"role": persona, "content": content})
            continue
        line = f'{label}: {content}'
        if messages and messages[-1]["role"] == 'user':
            messages[-1]["content"] += f'\n{line}'
        else:
            messages.append({"role": 'user', "content": line})
    return messages
