"""
API key authentication (H6).

Trước đây MỌI endpoint đều mở, kể cả `PUT /api/thresholds` (đổi ngưỡng cảnh báo
buồn ngủ) và `GET /api/events` (log hành trình + GPS của tài xế).

Thiết kế:
  - Không set env `GUARDIANPILOT_API_KEY` → auth TẮT, hành vi y hệt trước
    (demo local, pytest và CI không bị ảnh hưởng), chỉ log cảnh báo một lần.
  - Có set → các endpoint nhạy cảm yêu cầu header `X-API-Key`.

So sánh key bằng `hmac.compare_digest` để tránh timing attack.
"""
from __future__ import annotations

import functools
import hmac
import logging
import os

from flask import jsonify, request

logger = logging.getLogger(__name__)

ENV_VAR = "GUARDIANPILOT_API_KEY"
HEADER_NAME = "X-API-Key"

_warned_open = False


def get_api_key() -> str:
    return (os.getenv(ENV_VAR) or "").strip()


def auth_enabled() -> bool:
    return bool(get_api_key())


def _warn_open_mode_once() -> None:
    global _warned_open
    if not _warned_open:
        _warned_open = True
        logger.warning(
            "%s chưa được set — các endpoint nhạy cảm (thresholds/events/metrics) "
            "đang mở công khai. Chỉ dùng như vậy trên localhost.",
            ENV_VAR,
        )


def request_is_authorized() -> bool:
    """True nếu auth đang tắt, hoặc client gửi đúng key."""
    expected = get_api_key()
    if not expected:
        _warn_open_mode_once()
        return True
    provided = request.headers.get(HEADER_NAME) or request.args.get("api_key") or ""
    return hmac.compare_digest(str(provided), expected)


def require_api_key(view):
    """Decorator bảo vệ một route Flask."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not request_is_authorized():
            return jsonify({
                "ok": False,
                "error": f"Unauthorized — thiếu hoặc sai header {HEADER_NAME}.",
            }), 401
        return view(*args, **kwargs)

    return wrapper


def auth_status() -> dict:
    """Dùng cho /api/status — KHÔNG bao giờ trả về giá trị key."""
    return {"auth_required": auth_enabled(), "auth_header": HEADER_NAME}
