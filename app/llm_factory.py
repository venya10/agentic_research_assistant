"""Returns a LangChain chat model configured from .env."""

from app.config import cfg


def get_llm(temperature: float = 0.2):
    provider = cfg.LLM_PROVIDER

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=cfg.ANTHROPIC_MODEL,
            anthropic_api_key=cfg.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=4096,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg.OPENAI_MODEL,
            openai_api_key=cfg.OPENAI_API_KEY,
            temperature=temperature,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=cfg.GOOGLE_MODEL,
            google_api_key=cfg.GOOGLE_API_KEY,
            temperature=temperature,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=cfg.GROQ_MODEL,
            groq_api_key=cfg.GROQ_API_KEY,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
