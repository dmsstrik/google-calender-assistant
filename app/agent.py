from typing import Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool

from app import calendar_tools
from app.chains import build_intent_chain
from app.llm import build_llm
from app.prompts import agent_prompt


def _describe_event(event):
    return calendar_tools.format_events([event]).splitlines()[0]


# ===================== LIST =====================

@tool
def gcal_list(
    time_min_iso: str = "",
    time_max_iso: str = "",
    max_results: int = 10,
) -> str:
    """
    Показать список событий в произвольном диапазоне.
    Используйте ISO-формат времени (например 2025-12-24T09:00+03:00).
    """
    _mark_tool_called()
    try:
        items = calendar_tools.list_events(
            time_min_iso or None,
            time_max_iso or None,
            max_results=max_results,
        )
    except Exception as exc:
        return f"Ошибка при запросе списка: {exc}"
    return calendar_tools.format_events(items)


@tool
def gcal_list_today(
    day_label: str = "сегодня",
    max_results: int = 20,
) -> str:
    """Показать события за конкретный день (по умолчанию — сегодня)."""
    _mark_tool_called()
    try:
        items = calendar_tools.list_events_for_day(
            day_label or None,
            max_results=max_results,
        )
    except Exception as exc:
        return f"Ошибка при запросе событий за день: {exc}"
    return calendar_tools.format_events(items)


@tool
def gcal_list_period(
    start_label: str,
    end_label: str,
    max_results: int = 50,
) -> str:
    """
    Показать события за период.
    Форматы: YYYY-MM-DD, YYYY-MM-DDTHH:MM или «сегодня», «завтра».
    """
    _mark_tool_called()
    try:
        items = calendar_tools.list_events_for_period(
            start_label,
            end_label,
            max_results=max_results,
        )
    except Exception as exc:
        return f"Ошибка при запросе событий за период: {exc}"
    return calendar_tools.format_events(items)


# ===================== CREATE / UPDATE =====================

@tool
def gcal_create(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
) -> str:
    """Создать событие."""
    _mark_tool_called()
    try:
        event = calendar_tools.create_event(
            summary,
            start_iso,
            end_iso,
            description,
            location,
        )
    except Exception as exc:
        return f"Ошибка при создании события: {exc}"
    link = event.get("htmlLink") or "(без ссылки)"
    return f"Создано событие: {_describe_event(event)} | {link}"


@tool
def gcal_update(
    event_id: str,
    summary: str = "",
    start_iso: str = "",
    end_iso: str = "",
    description: str = "",
    location: str = "",
) -> str:
    """Обновить событие по event_id (любые поля опциональны)."""
    _mark_tool_called()
    payload = {}
    if summary:
        payload["summary"] = summary
    if start_iso:
        payload["start_iso"] = start_iso
    if end_iso:
        payload["end_iso"] = end_iso
    if description:
        payload["description"] = description
    if location:
        payload["location"] = location
    try:
        event = calendar_tools.update_event(event_id, **payload)
    except Exception as exc:
        return f"Ошибка при обновлении события: {exc}"
    return f"Событие обновлено: {_describe_event(event)}"


# ===================== DELETE =====================

@tool
def gcal_delete(event_id: str) -> str:
    """Удалить событие по известному event_id."""
    _mark_tool_called()
    try:
        calendar_tools.delete_event(event_id)
    except Exception as exc:
        return f"Ошибка при удалении события: {exc}"
    return f"Событие с id={event_id} удалено."


@tool
def gcal_safe_delete(
    time_min_iso: str = "",
    time_max_iso: str = "",
    day_label: str = "",
    start_label: str = "",
    end_label: str = "",
    around_label: str = "",
    summary: str = "",
) -> str:
    """
    _mark_tool_called()
    Удаление при неизвестном event_id. Можно передать ISO-диапазон либо
    натуральный день/время + опциональное название события.
    """
    try:
        result = calendar_tools.delete_event_by_hint(
            summary=summary or None,
            time_min_iso=time_min_iso or None,
            time_max_iso=time_max_iso or None,
            day_label=day_label or None,
            start_label=start_label or None,
            end_label=end_label or None,
            around_label=around_label or None,
        )
    except Exception as exc:
        return f"Ошибка при удалении события без id: {exc}"

    status = result.get("status")
    if status == "deleted":
        return f"Удалено событие: {_describe_event(result['event'])}"
    if status == "ambiguous":
        lines = calendar_tools.format_events(result["events"])
        return (
            "Найдено несколько событий. Уточните event_id, полное название или время:\n"
            f"{lines}"
        )
    return "По указанным условиям событий не найдено."


# ===================== AGENT =====================

_ACTION_INTENTS = {"update", "delete"}
_SUCCESS_KEYWORDS = (
    "успеш",
    "создан",
    "создал",
    "добав",
    "удален",
    "удалён",
    "обнов",
    "готов",
    "выполн",
    "сделан",
    "получилось",
    "удалось",
    "success",
)

_TOOL_CALL_FLAG = {"called": False}


def _reset_tool_flag() -> None:
    _TOOL_CALL_FLAG["called"] = False


def _mark_tool_called() -> None:
    _TOOL_CALL_FLAG["called"] = True


def _looks_like_success(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in _SUCCESS_KEYWORDS)


class CalendarAgent:
    def __init__(
        self,
        executor: AgentExecutor,
        memory: ConversationBufferMemory,
        intent_chain,
    ) -> None:
        self._executor = executor
        self._memory = memory
        self._intent_chain = intent_chain

    def _detect_intent(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return "other"
        try:
            intent = self._intent_chain.invoke({"input": cleaned})
        except Exception:
            return "other"
        intent = (intent or "").strip().lower()
        if intent not in {"list", "create", "update", "delete", "other"}:
            return "other"
        return intent

    def _rollback_last_turn(self) -> None:
        chat_memory = getattr(self._memory, "chat_memory", None)
        if not chat_memory or not getattr(chat_memory, "messages", None):
            return
        if chat_memory.messages and getattr(chat_memory.messages[-1], "type", None) == "ai":
            chat_memory.messages.pop()

    def _replace_last_ai_message(self, text: str) -> None:
        self._rollback_last_turn()
        chat_memory = getattr(self._memory, "chat_memory", None)
        if chat_memory:
            chat_memory.add_ai_message(text)

    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        user_text = inputs.get("input", "")
        intent = self._detect_intent(user_text)
        _reset_tool_flag()
        result = self._executor.invoke(inputs, **kwargs)
        steps = result.get("intermediate_steps") or []
        if intent in _ACTION_INTENTS and not steps and not _TOOL_CALL_FLAG["called"]:
            output_text = result.get("output", "")
            if intent == "delete":
                target_event = calendar_tools.resolve_delete_candidate(user_text)
                if target_event:
                    try:
                        calendar_tools.delete_event(target_event["id"])
                    except Exception as exc:
                        failure_msg = f"Не удалось удалить событие автоматически: {exc}"
                        self._replace_last_ai_message(failure_msg)
                        return {"output": failure_msg}
                    calendar_tools.clear_cached_delete_candidates()
                    success_msg = f"Удалено событие: {_describe_event(target_event)}"
                    self._replace_last_ai_message(success_msg)
                    return {"output": success_msg}
            if intent == "update":
                target_event = calendar_tools.resolve_known_event(user_text)
                if target_event:
                    payload = calendar_tools.extract_update_payload_from_text(
                        user_text, target_event
                    )
                    if payload:
                        try:
                            event = calendar_tools.update_event(
                                target_event["id"], **payload
                            )
                        except Exception as exc:
                            failure_msg = f"Не удалось обновить событие автоматически: {exc}"
                            self._replace_last_ai_message(failure_msg)
                            return {"output": failure_msg}
                        success_msg = f"Событие обновлено: {_describe_event(event)}"
                        self._replace_last_ai_message(success_msg)
                        return {"output": success_msg}
            if _looks_like_success(output_text):
                failure_msg = (
                    "Не удалось выполнить команду автоматически. "
                    "Укажи название, дату/время или event_id, чтобы я смог вызвать нужный инструмент."
                )
                self._replace_last_ai_message(failure_msg)
                return {"output": failure_msg}
        return result

def build_agent():
    llm = build_llm()

    tools = [
        gcal_list,
        gcal_list_today,
        gcal_list_period,
        gcal_create,
        gcal_update,
        gcal_delete,
        gcal_safe_delete,
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )

    agent_llm = create_tool_calling_agent(
        llm,
        tools,
        agent_prompt,
    )

    executor = AgentExecutor(
        agent=agent_llm,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        verbose=True,
    )

    intent_chain = build_intent_chain(llm)

    return CalendarAgent(executor, memory, intent_chain)
