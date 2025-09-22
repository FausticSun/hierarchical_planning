import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

load_dotenv()

gpt_llm = AzureChatOpenAI(
    api_key=SecretStr(os.environ["GPT_API_KEY"]),
    azure_endpoint=os.environ["GPT_AZURE_ENDPOINT"],
    api_version="2025-01-01-preview",
    model="gpt-4o",
)

claude_llm = ChatOpenAI(
    api_key=SecretStr(os.environ["CLAUDE_API_KEY"]),
    base_url=os.environ["CLAUDE_BASE_URL"],
    model="databricks-claude-3-7-sonnet",
)
