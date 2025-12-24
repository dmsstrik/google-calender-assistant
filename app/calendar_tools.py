import base64
import datetime as dt
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from dateutil import parser as date_parser
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import settings

SCOPES = ["https://www.googleapis.com/auth/calendar"]
LOCAL_TZ = (
    dt.datetime.now(dt.timezone.utc).astimezone().tzinfo or dt.timezone.utc
)
DAY_ALIASES = {
    "сегодня": 0,
    "сегодняшний день": 0,
    "today": 0,
    "текущий день": 0,
    "tomorrow": 1,
    "завтра": 1,
    "послезавтра": 2,
    "после завтра": 2,
    "day after tomorrow": 2,
    "вчера": -1,
    "yesterday": -1,
    "позавчера": -2,
}

RUS_MONTH_MAP = {
    "январь": "january",
    "января": "january",
    "янв": "january",
    "февраль": "february",
    "февраля": "february",
    "фев": "february",
    "март": "march",
    "марта": "march",
    "мар": "march",
    "апрель": "april",
    "апреля": "april",
    "апр": "april",
    "май": "may",
    "мая": "may",
    "июнь": "june",
    "июня": "june",
    "июн": "june",
    "июль": "july",
    "июля": "july",
    "июл": "july",
    "август": "august",
    "августа": "august",
    "авг": "august",
    "сентябрь": "september",
    "сентября": "september",
    "сент": "september",
    "сен": "september",
    "октябрь": "october",
    "октября": "october",
    "окт": "october",
    "ноябрь": "november",
    "ноября": "november",
    "ноя": "november",
    "декабрь": "december",
    "декабря": "december",
    "дек": "december",
}

RUS_MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(RUS_MONTH_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

RU_MONTHS_GENITIVE = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]

DEFAULT_SEARCH_TOLERANCE_MINUTES = 120
GOOGLE_API_RETRY_ATTEMPTS = 3
GOOGLE_API_RETRY_DELAY = 1.0
DELETE_CANDIDATE_TTL_SECONDS = 300
KNOWN_EVENTS_TTL_SECONDS = 900
MAX_KNOWN_EVENTS = 20

_PENDING_DELETE_EVENTS: List[Dict[str, Any]] = []
_PENDING_DELETE_TIMESTAMP: float = 0.0
_KNOWN_EVENTS: List[Tuple[float, Dict[str, Any]]] = []


# ===================== AUTH =====================

def _get_creds() -> Credentials:
    try:
        return Credentials.from_authorized_user_file(
            settings.google_token_file, SCOPES
        )
    except Exception:
        flow = InstalledAppFlow.from_client_secrets_file(
            settings.google_client_secret_file, SCOPES
        )
        creds = flow.run_local_server(port=0)
        with open(settings.google_token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        return creds


def _service():
    return build("calendar", "v3", credentials=_get_creds())


# ===================== HELPERS =====================

def _normalize_datetime(value: dt.datetime) -> dt.datetime:
    """Ensure the datetime is timezone-aware and converted to LOCAL_TZ."""
    if value.tzinfo is None:
        return value.replace(tzinfo=LOCAL_TZ)
    return value.astimezone(LOCAL_TZ)


def _looks_like_date_only(raw: str) -> bool:
    text = raw.strip()
    if not text:
        return False
    if "T" in text or ":" in text:
        return False
    separators = {"-", ".", "/"}
    return any(sep in text for sep in separators)


def _normalize_month_names(raw: str) -> str:
    if not raw or not raw.strip():
        return raw

    def repl(match: re.Match[str]) -> str:
        key = match.group(0).lower()
        return RUS_MONTH_MAP.get(key, match.group(0))

    return RUS_MONTH_PATTERN.sub(repl, raw)


def _parse_day_token(raw: Optional[str]) -> dt.date:
    if raw is None or not raw.strip():
        return _normalize_datetime(dt.datetime.now()).date()
    token = raw.strip().lower()
    today = _normalize_datetime(dt.datetime.now()).date()
    if token in DAY_ALIASES:
        return today + dt.timedelta(days=DAY_ALIASES[token])
    try:
        parsed = date_parser.parse(
            _normalize_month_names(raw), dayfirst=True, fuzzy=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Не удалось распознать дату '{raw}'. "
            "Укажите день явно, например 2025-01-30."
        ) from exc
    return _normalize_datetime(parsed).date()


def _parse_datetime_token(raw: str, prefer_end: bool = False) -> dt.datetime:
    raw = raw.strip()
    if not raw:
        raise ValueError("Нужно указать дату или дату с временем.")
    token = raw.lower()
    if token in DAY_ALIASES:
        base_day = _parse_day_token(raw)
        time_value = dt.time.max if prefer_end else dt.time.min
        return dt.datetime.combine(base_day, time_value, tzinfo=LOCAL_TZ)
    try:
        parsed = date_parser.parse(
            _normalize_month_names(raw), dayfirst=True, fuzzy=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Не удалось распознать дату/время '{raw}'. "
            "Используйте ISO-формат, например 2025-01-30T10:00."
        ) from exc
    parsed = _normalize_datetime(parsed)
    if _looks_like_date_only(raw):
        time_value = dt.time.max if prefer_end else dt.time.min
        return dt.datetime.combine(parsed.date(), time_value, tzinfo=LOCAL_TZ)
    return parsed


def _coerce_to_rfc3339(raw: str, field_name: str) -> str:
    """
    Convert a user-provided ISO-like datetime string into a RFC3339 string
    with seconds precision and an explicit timezone offset.
    """
    if not raw or not raw.strip():
        raise ValueError(f"Empty datetime value for {field_name}.")
    normalized = raw.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = date_parser.isoparse(normalized)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            f"Invalid datetime '{raw}' for {field_name}. "
            "Use ISO format like 2025-02-14T10:00:00+03:00."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.isoformat(timespec="seconds")


def day_range(day_label: Optional[str] = None) -> Tuple[str, str]:
    """Return ISO start/end boundaries for a calendar day."""
    day = _parse_day_token(day_label)
    start = dt.datetime.combine(day, dt.time.min, tzinfo=LOCAL_TZ)
    end = start + dt.timedelta(days=1)
    return start.isoformat(), end.isoformat()


def period_range(start_label: str, end_label: str) -> Tuple[str, str]:
    """Return ISO boundaries for an arbitrary period."""
    start = _parse_datetime_token(start_label, prefer_end=False)
    end = _parse_datetime_token(end_label, prefer_end=True)
    if end <= start:
        raise ValueError("Дата окончания должна быть позже даты начала.")
    return start.isoformat(), end.isoformat()


def _looks_like_event_id(value: str) -> bool:
    if not value:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(ch in allowed for ch in value)


def _maybe_decode_event_id(token: str) -> Optional[str]:
    clean = token.strip()
    if not clean:
        return None
    padding = "=" * (-len(clean) % 4)
    try:
        decoded = base64.urlsafe_b64decode(clean + padding).decode("utf-8")
    except Exception:
        return None
    decoded = decoded.strip()
    if not decoded:
        return None
    if " " in decoded:
        return decoded.split()[0]
    if _looks_like_event_id(decoded):
        return decoded
    return None


def _normalize_event_id(raw_id: str) -> str:
    if not raw_id or not raw_id.strip():
        raise ValueError("Пустой event_id.")
    candidate = raw_id.strip()
    if candidate.startswith("http"):
        parsed = urlparse(candidate)
        query = parse_qs(parsed.query)
        eid_values = query.get("eid")
        if eid_values:
            candidate = eid_values[0].strip()
    candidate = candidate.strip()
    decoded = _maybe_decode_event_id(candidate)
    if decoded:
        return decoded
    if " " in candidate:
        candidate = candidate.split()[0]
    return candidate


def _execute_with_retries(request, attempts: int = GOOGLE_API_RETRY_ATTEMPTS) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return request.execute()
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(GOOGLE_API_RETRY_DELAY)
    if last_exc:
        raise last_exc
    raise RuntimeError("Google API request failed without exception.")


def _extract_event_datetime(payload: Dict[str, Any], prefer_end: bool = False) -> Optional[dt.datetime]:
    if not payload:
        return None
    iso_value = payload.get("dateTime")
    date_only = payload.get("date")
    if iso_value:
        iso_value = iso_value.replace("Z", "+00:00")
        dt_value = dt.datetime.fromisoformat(iso_value)
        return _normalize_datetime(dt_value)
    if date_only:
        date_obj = dt.date.fromisoformat(date_only)
        time_value = dt.time.max if prefer_end else dt.time.min
        return dt.datetime.combine(date_obj, time_value, tzinfo=LOCAL_TZ)
    return None


def _build_event_match_tokens(event: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    start_dt = _extract_event_datetime(event.get("start", {}))
    end_dt = _extract_event_datetime(event.get("end", {}), prefer_end=True)

    def add_token(value: Optional[str]) -> None:
        if value:
            tokens.append(value.lower())

    if start_dt:
        add_token(start_dt.strftime("%H:%M"))
        add_token(str(start_dt.hour))
        add_token(f"{start_dt.hour:02d}")
        add_token(start_dt.strftime("%Y-%m-%d"))
        add_token(start_dt.strftime("%d.%m.%Y"))
        add_token(str(start_dt.day))
        add_token(f"{start_dt.day} {RU_MONTHS_GENITIVE[start_dt.month - 1]}")
    if end_dt:
        add_token(end_dt.strftime("%H:%M"))
        add_token(str(end_dt.hour))
        add_token(f"{end_dt.hour:02d}")
    if start_dt and end_dt:
        add_token(f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}")
        add_token(f"{start_dt.hour}-{end_dt.hour}")
    return [token for token in tokens if token]


SUMMARY_KEYWORDS = ("назв", "title", "переимен", "переименуй")
TIME_RANGE_PATTERNS = [
    re.compile(
        r"\b[сc]\s+(?P<start>[\d:\samp\.]+)\s+(?:до|по)\s+(?P<end>[\d:\samp\.]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<start>\d{1,2}:\d{2})\s*[-—]\s*(?P<end>\d{1,2}:\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<start>\d{1,2})\s*[-—]\s*(?P<end>\d{1,2})\b",
        re.IGNORECASE,
    ),
]
SINGLE_TIME_PATTERNS = [
    re.compile(r"\bв\s+(?P<time>[\d:\samp\.]+)", re.IGNORECASE),
    re.compile(
        r"\bна\s+(?P<time>\d{1,2}(:\d{2})?\s*(?:am|pm)?)\b",
        re.IGNORECASE,
    ),
]
QUOTED_TEXT_PATTERN = re.compile(r"[\"'«“](.+?)[\"'»”]")


def _extract_new_summary(text: str) -> Optional[str]:
    lowered = text.lower()
    if not any(keyword in lowered for keyword in SUMMARY_KEYWORDS):
        return None
    quoted = QUOTED_TEXT_PATTERN.findall(text)
    if quoted:
        candidate = quoted[-1].strip()
        if candidate:
            return candidate
    if " на " in lowered:
        idx = lowered.rfind(" на ")
        candidate = text[idx + 4 :].strip().strip(".!?,")
        if candidate:
            return candidate
    return None


def _parse_time_with_reference(
    token: str, reference_dt: Optional[dt.datetime]
) -> Optional[dt.datetime]:
    if not token:
        return None
    normalized = token.strip()
    if not normalized:
        return None
    if reference_dt is None:
        reference_dt = _normalize_datetime(dt.datetime.now())
    base = reference_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if normalized.isdigit():
        normalized = f"{normalized}:00"
    try:
        parsed = date_parser.parse(normalized, dayfirst=True, default=base)
    except Exception:
        return None
    parsed = parsed.replace(
        year=base.year, month=base.month, day=base.day
    )
    return _normalize_datetime(parsed)


def _extract_time_range_from_text(
    text: str, reference_event: Dict[str, Any]
) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    reference_start = _extract_event_datetime(reference_event.get("start"))
    reference_end = _extract_event_datetime(
        reference_event.get("end"), prefer_end=True
    )
    duration = None
    if reference_start and reference_end:
        duration = reference_end - reference_start
    reference_point = reference_start or reference_end

    for pattern in TIME_RANGE_PATTERNS:
        match = pattern.search(text)
        if match:
            start_dt = _parse_time_with_reference(
                match.group("start"), reference_point
            )
            end_dt = _parse_time_with_reference(
                match.group("end"), reference_point
            )
            if start_dt and end_dt:
                return start_dt, end_dt

    for pattern in SINGLE_TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            start_dt = _parse_time_with_reference(
                match.group("time"), reference_point
            )
            if start_dt:
                if duration:
                    end_dt = start_dt + duration
                else:
                    end_dt = None
                return start_dt, end_dt

    return None, None


def extract_update_payload_from_text(
    user_text: str, reference_event: Dict[str, Any]
) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    if not user_text:
        return payload
    summary_value = _extract_new_summary(user_text)
    if summary_value:
        payload["summary"] = summary_value
    start_dt, end_dt = _extract_time_range_from_text(user_text, reference_event)
    if start_dt:
        payload["start_iso"] = start_dt.isoformat()
    if end_dt:
        payload["end_iso"] = end_dt.isoformat()
    return payload


def _cache_delete_candidates(events: List[Dict[str, Any]]) -> None:
    global _PENDING_DELETE_EVENTS, _PENDING_DELETE_TIMESTAMP
    _PENDING_DELETE_EVENTS = [event.copy() for event in events]
    _PENDING_DELETE_TIMESTAMP = time.time()


def clear_cached_delete_candidates() -> None:
    global _PENDING_DELETE_EVENTS, _PENDING_DELETE_TIMESTAMP
    _PENDING_DELETE_EVENTS = []
    _PENDING_DELETE_TIMESTAMP = 0.0


def resolve_cached_delete_candidate(user_text: str) -> Optional[Dict[str, Any]]:
    if not user_text:
        return None
    if not _PENDING_DELETE_EVENTS:
        return None
    if (
        _PENDING_DELETE_TIMESTAMP
        and time.time() - _PENDING_DELETE_TIMESTAMP > DELETE_CANDIDATE_TTL_SECONDS
    ):
        clear_cached_delete_candidates()
        return None

    lowered = user_text.lower()
    return _select_best_event(_PENDING_DELETE_EVENTS, lowered)


def remember_known_events(events: List[Dict[str, Any]], replace: bool = False) -> None:
    if not events:
        return
    now = time.time()
    global _KNOWN_EVENTS
    filtered = [
        (ts, ev)
        for ts, ev in _KNOWN_EVENTS
        if now - ts <= KNOWN_EVENTS_TTL_SECONDS
    ]
    if replace:
        filtered = []
    existing_index = {
        ev.get("id"): idx for idx, (_, ev) in enumerate(filtered) if ev.get("id")
    }
    for event in events:
        if not isinstance(event, dict):
            continue
        copy_event = event.copy()
        event_id = copy_event.get("id")
        entry = (now, copy_event)
        if event_id and event_id in existing_index:
            filtered[existing_index[event_id]] = entry
        else:
            filtered.append(entry)
    if len(filtered) > MAX_KNOWN_EVENTS:
        filtered = filtered[-MAX_KNOWN_EVENTS:]
    _KNOWN_EVENTS = filtered


def remember_single_event(event: Dict[str, Any]) -> None:
    remember_known_events([event], replace=False)


def forget_known_event(event_id: Optional[str]) -> None:
    if not event_id:
        return
    global _KNOWN_EVENTS
    _KNOWN_EVENTS = [
        (ts, ev)
        for ts, ev in _KNOWN_EVENTS
        if ev.get("id") != event_id
    ]


def resolve_known_event(user_text: str) -> Optional[Dict[str, Any]]:
    if not user_text:
        return None
    now = time.time()
    lowered = user_text.lower()
    events = [
        ev.copy()
        for ts, ev in _KNOWN_EVENTS
        if now - ts <= KNOWN_EVENTS_TTL_SECONDS
    ]
    return _select_best_event(events, lowered)


def resolve_delete_candidate(user_text: str) -> Optional[Dict[str, Any]]:
    event = resolve_cached_delete_candidate(user_text)
    if event:
        return event
    return resolve_known_event(user_text)


def _score_event_against_text(event: Dict[str, Any], lowered: str) -> int:
    score = 0
    event_id = (event.get("id") or "").lower()
    if event_id and event_id in lowered:
        score += 6
    summary = (event.get("summary") or "").lower()
    if summary:
        if summary in lowered:
            score += 4
        else:
            summary_words = [word for word in re.split(r"\s+", summary) if word]
            score += sum(1 for word in summary_words if word in lowered)
    for token in _build_event_match_tokens(event):
        if token in lowered:
            score += 1
    return score


def _select_best_event(
    events: List[Dict[str, Any]], lowered: str
) -> Optional[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for event in events:
        score = _score_event_against_text(event, lowered)
        if score > 0:
            scored.append((score, event))
    if not scored:
        if len(events) == 1:
            return events[0]
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    best = [event for score, event in scored if score == top_score]
    if len(best) == 1:
        return best[0]
    return None


def _parse_iso_datetime(raw: str, field_name: str) -> dt.datetime:
    """Parse an ISO-like string and normalize it to LOCAL_TZ."""
    if not raw or not raw.strip():
        raise ValueError(f"Empty value for {field_name}.")
    try:
        parsed = date_parser.isoparse(raw.strip())
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Cannot parse {field_name} '{raw}'.") from exc
    return _normalize_datetime(parsed)


def _resolve_time_window(
    time_min_iso: Optional[str] = None,
    time_max_iso: Optional[str] = None,
    day_label: Optional[str] = None,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    around_label: Optional[str] = None,
    tolerance_minutes: int = DEFAULT_SEARCH_TOLERANCE_MINUTES,
) -> Tuple[str, str]:
    """
    Derive a (start, end) ISO window from natural language hints.
    """
    tolerance_delta = dt.timedelta(minutes=max(tolerance_minutes, 1))

    if time_min_iso or time_max_iso:
        if time_min_iso and time_max_iso:
            return time_min_iso, time_max_iso
        if time_min_iso:
            start_dt = _parse_iso_datetime(time_min_iso, "time_min_iso")
            end_dt = start_dt + tolerance_delta
            return start_dt.isoformat(), end_dt.isoformat()
        end_dt = _parse_iso_datetime(time_max_iso, "time_max_iso")
        start_dt = end_dt - tolerance_delta
        return start_dt.isoformat(), end_dt.isoformat()

    if start_label and end_label:
        return period_range(start_label, end_label)

    if start_label:
        start_dt = _parse_datetime_token(start_label)
        end_dt = start_dt + tolerance_delta
        return start_dt.isoformat(), end_dt.isoformat()

    if day_label:
        return day_range(day_label)

    if end_label:
        end_dt = _parse_datetime_token(end_label, prefer_end=True)
        start_dt = end_dt - tolerance_delta
        return start_dt.isoformat(), end_dt.isoformat()

    if around_label:
        center = _parse_datetime_token(around_label)
        start_dt = center - tolerance_delta
        end_dt = center + tolerance_delta
        return start_dt.isoformat(), end_dt.isoformat()

    raise ValueError(
        "Укажи ISO-диапазон, day_label, start/end либо around_label, чтобы найти событие."
    )


def _humanize_event_time(payload: Dict[str, Any], label: str) -> str:
    iso_value = payload.get("dateTime")
    date_only = payload.get("date")
    if iso_value:
        iso_value = iso_value.replace("Z", "+00:00")
        dt_value = dt.datetime.fromisoformat(iso_value)
        dt_value = _normalize_datetime(dt_value)
        return dt_value.strftime("%Y-%m-%d %H:%M")
    if date_only:
        return f"{date_only} (весь день)"
    return f"[нет {label}]"


def format_events(events: List[Dict[str, Any]]) -> str:
    """Return a readable list of events."""
    if not events:
        return "События не найдены."

    lines = []
    for event in events:
        summary = event.get("summary") or "(без названия)"
        start = _humanize_event_time(event.get("start", {}), "начала")
        end = _humanize_event_time(event.get("end", {}), "окончания")
        location = event.get("location")
        line = f"- {start} -> {end} | {summary} | id={event.get('id')}"
        if location:
            line += f" | {location}"
        lines.append(line)
    return "\n".join(lines)


# ===================== CORE API =====================

def list_events(
    time_min_iso: Optional[str] = None,
    time_max_iso: Optional[str] = None,
    max_results: int = 10,
    calendar_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Получает события за указанный временной промежуток.
    Если рамки не заданы, возвращаются ближайшие события.
    """
    calendar_id = calendar_id or settings.default_calendar_id
    svc = _service()

    params = {
        "calendarId": calendar_id,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }

    if time_min_iso:
        params["timeMin"] = _coerce_to_rfc3339(time_min_iso, "time_min")
    if time_max_iso:
        params["timeMax"] = _coerce_to_rfc3339(time_max_iso, "time_max")

    request = svc.events().list(**params)
    events_result = _execute_with_retries(request)
    items = events_result.get("items", [])
    remember_known_events(items, replace=True)
    return items


def list_events_for_day(
    day_label: Optional[str] = None,
    max_results: int = 20,
    calendar_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Удобный шорткат для получения событий за конкретный день."""
    start, end = day_range(day_label)
    return list_events(start, end, max_results=max_results, calendar_id=calendar_id)


def list_events_for_period(
    start_label: str,
    end_label: str,
    max_results: int = 50,
    calendar_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Получает события за произвольный период."""
    start, end = period_range(start_label, end_label)
    return list_events(start, end, max_results=max_results, calendar_id=calendar_id)


def find_events_by_hint(
    summary: Optional[str] = None,
    time_min_iso: Optional[str] = None,
    time_max_iso: Optional[str] = None,
    day_label: Optional[str] = None,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    around_label: Optional[str] = None,
    tolerance_minutes: int = DEFAULT_SEARCH_TOLERANCE_MINUTES,
    max_results: int = 10,
    calendar_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Select events by mixing summary keywords with natural language time hints.
    """
    start_iso, end_iso = _resolve_time_window(
        time_min_iso=time_min_iso,
        time_max_iso=time_max_iso,
        day_label=day_label,
        start_label=start_label,
        end_label=end_label,
        around_label=around_label,
        tolerance_minutes=tolerance_minutes,
    )
    events = list_events(
        start_iso,
        end_iso,
        max_results=max_results,
        calendar_id=calendar_id,
    )
    if summary:
        needle = summary.strip().lower()
        if needle:
            events = [
                event
                for event in events
                if needle in (event.get("summary") or "").lower()
            ]
    return events


def delete_event_by_hint(
    summary: Optional[str] = None,
    time_min_iso: Optional[str] = None,
    time_max_iso: Optional[str] = None,
    day_label: Optional[str] = None,
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    around_label: Optional[str] = None,
    tolerance_minutes: int = DEFAULT_SEARCH_TOLERANCE_MINUTES,
    calendar_id: Optional[str] = None,
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Delete a single event by first locating it using natural language hints.
    Returns a dict with status deleted/not_found/ambiguous and matching events.
    """
    matches = find_events_by_hint(
        summary=summary,
        time_min_iso=time_min_iso,
        time_max_iso=time_max_iso,
        day_label=day_label,
        start_label=start_label,
        end_label=end_label,
        around_label=around_label,
        tolerance_minutes=tolerance_minutes,
        max_results=max_results,
        calendar_id=calendar_id,
    )
    if not matches:
        clear_cached_delete_candidates()
        return {"status": "not_found", "events": []}
    if len(matches) > 1:
        _cache_delete_candidates(matches)
        return {"status": "ambiguous", "events": matches}

    clear_cached_delete_candidates()
    target = matches[0]
    delete_event(target["id"], calendar_id=calendar_id)
    return {"status": "deleted", "event": target}


def create_event(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
    calendar_id: Optional[str] = None,
) -> Dict[str, Any]:
    calendar_id = calendar_id or settings.default_calendar_id
    svc = _service()

    start_iso = _coerce_to_rfc3339(start_iso, "start")
    end_iso = _coerce_to_rfc3339(end_iso, "end")

    body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }

    event = svc.events().insert(
        calendarId=calendar_id,
        body=body,
    ).execute()
    remember_single_event(event)
    return event


def update_event(
    event_id: str,
    summary: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: Optional[str] = None,
) -> Dict[str, Any]:
    calendar_id = calendar_id or settings.default_calendar_id
    svc = _service()

    event_id = _normalize_event_id(event_id)

    event = _execute_with_retries(
        svc.events().get(
            calendarId=calendar_id,
            eventId=event_id,
        )
    )

    if summary is not None:
        event["summary"] = summary
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    if start_iso is not None:
        event["start"] = {
            "dateTime": _coerce_to_rfc3339(start_iso, "start")
        }
    if end_iso is not None:
        event["end"] = {"dateTime": _coerce_to_rfc3339(end_iso, "end")}

    updated = _execute_with_retries(
        svc.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event,
        )
    )
    remember_single_event(updated)
    return updated


def delete_event(
    event_id: str,
    calendar_id: Optional[str] = None,
) -> Dict[str, Any]:
    calendar_id = calendar_id or settings.default_calendar_id
    svc = _service()

    event_id = _normalize_event_id(event_id)

    _execute_with_retries(
        svc.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
        )
    )
    forget_known_event(event_id)

    return {"status": "deleted", "event_id": event_id}
