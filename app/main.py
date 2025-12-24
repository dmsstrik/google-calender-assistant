import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console

from app.agent import build_agent
from app.config import settings

console = Console()


def run_cli():
    agent = build_agent()
    console.print(
        "[bold]Ассистент Google Calendar (CLI). Напишите команду или 'exit' для выхода.[/bold]"
    )
    while True:
        try:
            user_text = console.input("\n[cyan]>>> [/cyan]").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                console.print("[yellow]Пустой запрос. Повторите попытку.[/yellow]")
                continue

            result = agent.invoke({"input": user_text})
            console.print(f"[green]Ответ:[/green] {result['output']}")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            console.print(f"[red]Ошибка:[/red] {exc}")


if __name__ == "__main__":
    if settings.interface == "cli":
        run_cli()
    else:
        raise RuntimeError(
            "Поддерживается только CLI-интерфейс. Измените переменную окружения INTERFACE."
        )
