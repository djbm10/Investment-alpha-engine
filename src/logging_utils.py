from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, object]) -> tuple[str, dict[str, object]]:
        extra = kwargs.setdefault("extra", {})
        context = extra.setdefault("context", {})
        if isinstance(context, dict):
            context.update(self.extra)
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        return json.dumps(payload, default=str)


def setup_logger(log_file: Path, **context: object) -> ContextAdapter:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"phase1_pipeline:{log_file}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return ContextAdapter(logger, context)
