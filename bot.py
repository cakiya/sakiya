"""Entry point. Boots the local model server, warms the RAG index, then runs the bot."""

from sakiya import prompts, rag, server
from sakiya.client import bot
from sakiya.settings import settings


def main():
    server.start_server(str(settings.kobold_exe), str(settings.model_path), settings.kobold_port)
    rag.build_index()
    prompts.load_system_prompt()
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
