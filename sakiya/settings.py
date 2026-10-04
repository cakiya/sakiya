"""Central configuration. Every tunable the bot uses is named here exactly once."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

FALLBACK_SYSTEM_PROMPT = "You are sakiya."


@dataclass(frozen=True)
class DrySampling:
    """KoboldCpp dry-run sampler, passed through to the OpenAI-compatible endpoint."""

    multiplier: float = 0.8
    base: float = 1.75
    allowed_length: int = 2
    penalty_last_n: int = 1024


@dataclass(frozen=True)
class Settings:
    discord_token: str
    model_name: str
    prompt_name: str

    kobold_exe: Path
    model_path: Path
    prompt_path: Path
    personas_dir: Path

    llm_url: str
    kobold_port: str
    llm_model: str
    command_prefix: str

    max_channel_memory: int
    max_user_memory: int

    embedding_model: str
    embedding_local_only: bool
    rag_top_k: int
    rag_chunk_chars: int

    temperature: float
    frequency_penalty: float
    presence_penalty: float
    min_p: float
    top_p: float
    rep_pen: float
    banned_strings: tuple[str, ...]
    dry: DrySampling

    send_chunk_size: int


def load_settings() -> Settings:
    model_name = os.getenv("MODEL_NAME", "default")
    prompt_name = os.getenv("PROMPT", "default")
    port = os.getenv("KOBOLD_PORT", "5001")

    return Settings(
        discord_token=os.getenv("DISCORD_BOT_TOKEN", ""),
        model_name=model_name,
        prompt_name=prompt_name,
        kobold_exe=BASE_DIR / "koboldcpp.exe",
        model_path=BASE_DIR / "input" / "models" / f"{model_name}.gguf",
        prompt_path=BASE_DIR / "input" / "prompts" / f"{prompt_name}.txt",
        personas_dir=BASE_DIR / "input" / "personas",
        llm_url=f"http://localhost:{port}/v1",
        kobold_port=port,
        llm_model="local-model",
        command_prefix="!",
        max_channel_memory=100,
        max_user_memory=10,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_local_only=True,
        rag_top_k=5,
        rag_chunk_chars=80,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        min_p=0.05,
        top_p=1.0,
        rep_pen=1.15,
        banned_strings=(
            "<" + "br" + ">",
            "<" + "BR" + ">",
            "<" + "/br" + ">",
            "<" + "/BR" + ">",
            "<" + "br/" + ">",
            "<" + "br /" + ">"
        ),
        dry=DrySampling(),
        send_chunk_size=1950,
    )


settings = load_settings()
