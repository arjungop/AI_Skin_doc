import os
from typing import List, Dict, Optional
import json
import requests


def _provider() -> Optional[str]:
    # Optional explicit override
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced in {"gemini", "azure", "openai", "ollama"}:
        return forced
    # Priority: Gemini -> Azure OpenAI -> Ollama
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        return "azure"
    if os.getenv("OLLAMA_BASE_URL") and os.getenv("OLLAMA_MODEL"):
        return "ollama"
    # Keep OpenAI as last fallback
    if os.getenv("OPENAI_API_KEY"):
        os.environ.setdefault("OPENAI_MODEL", os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"))
        return "openai"
    return None


def _gemini_chat(messages: List[Dict[str, str]]) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai package not installed. Add 'google-generativeai>=0.3.0' to requirements.") from e

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # Use the correct model name for the current API
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        # Gemini expects alternating user/model messages, so we need to convert system + history
        prompt_parts = []
        system_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content and not prompt_parts:
                    # Add system message as initial context
                    prompt_parts.append(f"System Instructions: {system_content}\n\nUser: {msg['content']}")
                    system_content = ""
                else:
                    prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        # Combine all parts into a single prompt
        full_prompt = "\n\n".join(prompt_parts)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_output_tokens=2048,
        )
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"[LLM error: Gemini] {e}"


def _gemini_chat_stream(messages: List[Dict[str, str]]):
    try:
        import google.generativeai as genai
    except Exception as e:
        yield "[LLM error: Gemini streaming not available]"
        return

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # Use the correct model name for the current API
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # Convert messages to Gemini format
        prompt_parts = []
        system_content = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content and not prompt_parts:
                    prompt_parts.append(f"System Instructions: {system_content}\n\nUser: {msg['content']}")
                    system_content = ""
                else:
                    prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_output_tokens=2048,
        )
        
        # Use streaming response
        response = model.generate_content(
            full_prompt, 
            generation_config=generation_config,
            stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"[LLM error: Gemini] {e}"


def _azure_chat(messages: List[Dict[str, str]]) -> str:
    try:
        from openai import AzureOpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Add 'openai>=1.0.0' to requirements.") from e

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM error: Azure OpenAI] {e}"


def _openai_chat(messages: List[Dict[str, str]]) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Add 'openai>=1.0.0' to requirements.") from e
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM error: OpenAI] {e}"


def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    base = os.getenv("OLLAMA_BASE_URL").rstrip("/")
    model = os.getenv("OLLAMA_MODEL")
    url = f"{base}/api/chat"
    # Map performance and safety knobs from env → Ollama options
    opts = {}
    if os.getenv("OLLAMA_NUM_PREDICT"):
        try: opts["num_predict"] = int(os.getenv("OLLAMA_NUM_PREDICT"))
        except Exception: pass
    if os.getenv("OLLAMA_NUM_CTX"):
        try: opts["num_ctx"] = int(os.getenv("OLLAMA_NUM_CTX"))
        except Exception: pass
    if os.getenv("OLLAMA_TEMPERATURE"):
        try: opts["temperature"] = float(os.getenv("OLLAMA_TEMPERATURE"))
        except Exception: pass
    if os.getenv("OLLAMA_NUM_THREADS"):
        try: opts["num_thread"] = int(os.getenv("OLLAMA_NUM_THREADS"))
        except Exception: pass
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": opts or None,
        "keep_alive": keep_alive,
    }
    try:
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
        r = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"[LLM error: Ollama] {e}"
    # Ollama chat returns {'message': {'role': 'assistant', 'content': '...'}}
    if isinstance(data, dict):
        msg = data.get("message") or {}
        content = msg.get("content")
        if content:
            return content.strip()
    # Fallback try OpenAI-like structure
    if "choices" in data:
        return (data["choices"][0]["message"]["content"] or "").strip()
    return ""


# ---------------- Streaming helpers -----------------
def _openai_chat_stream(messages: List[Dict[str, str]]):
    try:
        from openai import OpenAI
    except Exception as e:
        yield "[LLM error: OpenAI streaming not available]"
        return
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                yield delta.content
    except Exception as e:
        yield f"[LLM error: OpenAI] {e}"


def _azure_chat_stream(messages: List[Dict[str, str]]):
    try:
        from openai import AzureOpenAI
    except Exception as e:
        yield "[LLM error: Azure streaming not available]"
        return
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    try:
        stream = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            stream=True,
        )
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                yield delta.content
    except Exception as e:
        yield f"[LLM error: Azure] {e}"


def _ollama_chat_stream(messages: List[Dict[str, str]]):
    base = os.getenv("OLLAMA_BASE_URL").rstrip("/")
    model = os.getenv("OLLAMA_MODEL")
    url = f"{base}/api/chat"
    opts = {}
    if os.getenv("OLLAMA_NUM_PREDICT"):
        try: opts["num_predict"] = int(os.getenv("OLLAMA_NUM_PREDICT"))
        except Exception: pass
    if os.getenv("OLLAMA_NUM_CTX"):
        try: opts["num_ctx"] = int(os.getenv("OLLAMA_NUM_CTX"))
        except Exception: pass
    if os.getenv("OLLAMA_TEMPERATURE"):
        try: opts["temperature"] = float(os.getenv("OLLAMA_TEMPERATURE"))
        except Exception: pass
    if os.getenv("OLLAMA_NUM_THREADS"):
        try: opts["num_thread"] = int(os.getenv("OLLAMA_NUM_THREADS"))
        except Exception: pass
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": opts or None,
        "keep_alive": keep_alive,
    }
    try:
        with requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, stream=True, timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))) as r:
            r.raise_for_status()
            last_len = 0
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                # Try 'message.content' cumulative
                msg = (data.get("message") or {})
                content = msg.get("content")
                if content is not None:
                    new = content[last_len:]
                    last_len = len(content)
                    if new:
                        yield new
                    continue
                # Or 'response' incremental
                if "response" in data and data["response"]:
                    yield str(data["response"])
                if data.get("done"):
                    break
    except Exception as e:
        yield f"[LLM error: Ollama] {e}"


def stream_chat_reply(prompt: str, patient: Optional[Dict] = None, history: Optional[List[Dict[str, str]]] = None):
    # Build messages like chat_reply
    system = {
        "role": "system",
        "content": (
            "You are 'AI Skin Doctor', a friendly medical information assistant. "
            "Your job is to EDUCATE users about skin health: provide symptom lists, red flags, common causes, self‑care tips, and guidance on when to seek urgent care. "
            "You MUST NOT provide a definitive diagnosis or prescribe medication. Always include a short disclaimer to consult a clinician. "
            "If asked about melanoma or suspicious moles, include the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution) and the 'ugly duckling' sign. "
            "Be practical, concise, and supportive. \n\n"
            "Format your answer in Markdown with these sections: \n"
            "1) Summary (2–3 sentences)\n"
            "2) Symptoms/Features — bullet list (max 6)\n"
            "3) Red Flags — bullet list (max 6)\n"
            "4) Self‑Care — bullet list (max 6)\n"
            "5) When to Seek Care — bullet list (max 6)\n"
            "6) Disclaimer — one sentence."
        ),
    }
    context_parts = []
    if patient:
        context_parts.append(
            f"Patient context: name={patient.get('first_name','') or ''} {patient.get('last_name','') or ''}, "
            f"age={patient.get('age','?')}, gender={patient.get('gender','?')}"
        )
    if context_parts:
        system["content"] += "\n" + "\n".join(context_parts)
    msgs: List[Dict[str, str]] = [system]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})

    prov = _provider()
    if prov == "gemini":
        return _gemini_chat_stream(msgs)
    if prov == "azure":
        return _azure_chat_stream(msgs)
    if prov == "openai":
        return _openai_chat_stream(msgs)
    if prov == "ollama":
        return _ollama_chat_stream(msgs)
    # Fallback single chunk
    def _iter():
        yield "LLM not configured."
    return _iter()


def chat_reply(prompt: str, patient: Optional[Dict] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate a chat reply from the LLM. History is a list of {role, content}.
    Patient may contain demographic/context fields.
    """
    system = {
        "role": "system",
        "content": (
            "You are 'AI Skin Doctor', a friendly medical information assistant. "
            "Your job is to EDUCATE users about skin health: provide symptom lists, red flags, common causes, self‑care tips, and guidance on when to seek urgent care. "
            "You MUST NOT provide a definitive diagnosis or prescribe medication. Always include a short disclaimer to consult a clinician. "
            "If asked about melanoma or suspicious moles, include the ABCDE rule and the 'ugly duckling' sign. "
            "Be practical, concise, and supportive. \n\n"
            "Format your answer in Markdown with these sections: \n"
            "1) Summary (2–3 sentences)\n"
            "2) Symptoms/Features — bullet list (max 6)\n"
            "3) Red Flags — bullet list (max 6)\n"
            "4) Self‑Care — bullet list (max 6)\n"
            "5) When to Seek Care — bullet list (max 6)\n"
            "6) Disclaimer — one sentence."
        ),
    }
    context_parts = []
    if patient:
        context_parts.append(
            f"Patient context: name={patient.get('first_name','') or ''} {patient.get('last_name','') or ''}, "
            f"age={patient.get('age','?')}, gender={patient.get('gender','?')}"
        )
    if context_parts:
        system["content"] += "\n" + "\n".join(context_parts)

    msgs: List[Dict[str, str]] = [system]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": prompt})

    prov = _provider()
    if prov == "gemini":
        return _gemini_chat(msgs)
    if prov == "azure":
        return _azure_chat(msgs)
    if prov == "openai":
        return _openai_chat(msgs)
    if prov == "ollama":
        return _ollama_chat(msgs)

    # Dev fallback (no LLM configured)
    return "I'm your AI Skin Doctor assistant. I don't have LLM access configured yet, but I can still share general guidance. Please consult a clinician for diagnosis."


def diagnosis_for_lesion(patient: Dict, lesion: Dict) -> str:
    """Generate a patient-friendly explanation and next steps based on lesion info."""
    system = {
        "role": "system",
        "content": (
            "You are an AI assistant helping explain a skin lesion classification to a patient. "
            "Summarize in simple terms; include likely features, risk factors, red flags, safe self‑care, and clear next steps. "
            "ALWAYS include a short medical disclaimer and suggest appropriate professional follow‑up; do not give a definitive diagnosis or prescribe medication. \n\n"
            "Format in Markdown: Summary; What it means; Red Flags (bullets); Self‑Care (bullets); Follow‑Up (bullets); Disclaimer (one sentence)."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Patient: age={patient.get('age','?')}, gender={patient.get('gender','?')}.\n"
            f"Lesion: prediction={lesion.get('prediction','unknown')}, image={lesion.get('image_path','n/a')}.\n"
            "Provide: (1) one-paragraph explanation, (2) bullet list of warning signs, (3) suggested follow-up timeline."
        ),
    }
    messages = [system, user]

    prov = _provider()
    if prov == "gemini":
        return _gemini_chat(messages)
    if prov == "azure":
        return _azure_chat(messages)
    if prov == "openai":
        return _openai_chat(messages)
    if prov == "ollama":
        return _ollama_chat(messages)

    # Dev fallback
    pred = lesion.get("prediction", "unknown")
    return (
        f"Preliminary AI classification suggests: {pred}. This is not a medical diagnosis.\n\n"
        "Warning signs: rapid growth, irregular borders, multiple colors, bleeding, or pain.\n"
        "Follow-up: please consult a dermatologist within 1–2 weeks, or sooner if symptoms worsen."
    )
