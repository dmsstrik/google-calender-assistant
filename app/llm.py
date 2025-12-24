from langchain_openai import ChatOpenAI
from app.config import settings

def build_llm():
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0.2,
    )
