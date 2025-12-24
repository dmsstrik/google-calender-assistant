Google Calendar Assistant
=========================

Инструмент командной строки, который помогает управлять событиями Google Calendar с помощью LLM-агента на базе LangChain. Агент умеет распознавать намерения (просмотр событий, создание, обновление, удаление) и вызывает специализированные инструменты, которые работают через Google Calendar API.

Возможности
-----------
- CLI-интерфейс с подсказками и цветным выводом через `rich`.
- Интеграция с Google Calendar API для чтения и изменения событий.
- Управление конфигурацией через `.env` и `app/config.py` без внешних зависимостей.
- Набор инструментов (`app/calendar_tools.py`) для работы с периодами, днями, подсказками на естественном языке.

Требования
----------
- Python 3.10+
- Активированный Google Cloud проект с авторизованным OAuth client (файлы `client_secret.json` и `token.json`).
- Токен LLM (например, OpenAI API key или совместимый endpoint), если используется облачная модель.

Установка
---------
1. Клонируй репозиторий:
   ```bash
   git clone https://github.com/<user>/google-calender-assistant.git
   cd google-calender-assistant
   ```
2. Создай виртуальное окружение и установи зависимости:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

Настройка окружения
-------------------
1. Скопируй `.env.example` в `.env` и укажи значения переменных:
   ```
   OPENAI_BASE_URL=
   OPENAI_API_KEY=
   LLM_MODEL=qwen2.5-vl-7b-instruct
   INTERFACE=cli
   GOOGLE_CLIENT_SECRET_FILE=secrets/client_secret.json
   GOOGLE_TOKEN_FILE=secrets/token.json
   DEFAULT_CALENDAR_ID=primary
   TELEGRAM_BOT_TOKEN=
   ```
2. Помести файлы `client_secret.json` и `token.json` в папку `secrets/`. Они игнорируются Git и должны храниться только локально.

Запуск
------
```bash
python -m app.main
```
CLI попросит команду. Примеры:
- `покажи события на сегодня`
- `создай встречу завтра с 10 до 11 про созвон с дизайнером`
- `удали событие «дедлайн отчета»`

Публикация на GitHub
--------------------
1. Убедись, что в `.gitignore` перечислены `.env`, `secrets/`, `__pycache__/`.
2. Проверь локальные изменения и сделай коммит:
   ```bash
   git status
   git add .
   git commit -m "Добавить README и настройки"
   ```
3. Залей изменения в GitHub:
   ```bash
   git push origin main        # или нужную ветку
   ```
Если нужно заменить содержимое удалённой ветки полностью, используй `git push origin main --force` (только если уверен, что хочешь переписать историю).
