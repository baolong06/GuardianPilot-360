"""Thread-safe writer for annotated video analysis output files."""

from __future__ import annotations

import re
import secrets
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


class VideoOutputError(RuntimeError):
    """Raised when an output session cannot be created or updated."""


@dataclass
class _VideoSession:
    output_id: str
    filename: str
    path: Path
    writer: Any
    width: int
    height: int
    fps: float
    codec: str = "mp4v"
    frame_count: int = 0
    complete: bool = False
    updated_at: float = 0.0


class VideoOutputStore:
    """Owns active ``cv2.VideoWriter`` instances and completed outputs."""

    MAX_ACTIVE_SESSIONS = 4
    STALE_SESSION_SEC = 300.0

    # M10: thử H.264 trước — `mp4v` (MPEG-4 Part 2) tạo được file .mp4 nhưng
    # Chrome/Firefox/Safari KHÔNG phát được inline, người dùng tải về rồi mở
    # bằng trình xem ngoài mới thấy. `avc1` phát trực tiếp trên trình duyệt.
    # OpenCV chỉ có avc1 khi build kèm openh264/FFmpeg → phải có fallback.
    CODEC_PREFERENCE = ("avc1", "mp4v")

    def __init__(
        self,
        output_dir: str | Path,
        writer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self._writer_factory = writer_factory or cv2.VideoWriter
        self._sessions: dict[str, _VideoSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _safe_stem(original_name: str) -> str:
        stem = Path(original_name or "video").stem
        stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-.")
        return (stem or "video")[:64]

    @staticmethod
    def _validate_geometry(width: int, height: int, fps: float) -> tuple[int, int, float]:
        if not 16 <= width <= 3840 or not 16 <= height <= 2160:
            raise VideoOutputError("Video dimensions must be between 16px and 3840x2160.")
        if not 0.5 <= fps <= 60.0:
            raise VideoOutputError("Output FPS must be between 0.5 and 60.")
        # Most MP4 encoders require even frame dimensions.
        return width - (width % 2), height - (height % 2), fps

    def start(self, original_name: str, width: int, height: int, fps: float) -> dict:
        width, height, fps = self._validate_geometry(int(width), int(height), float(fps))
        output_id = secrets.token_urlsafe(18)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._safe_stem(original_name)}_analyzed_{stamp}_{output_id[:8]}.mp4"

        with self._lock:
            self._close_stale_sessions()
            active_count = sum(not item.complete for item in self._sessions.values())
            if active_count >= self.MAX_ACTIVE_SESSIONS:
                raise VideoOutputError("Too many active video output sessions.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = (self.output_dir / filename).resolve()
            if path.parent != self.output_dir:
                raise VideoOutputError("Invalid output path.")

            writer, codec = self._open_writer(path, fps, width, height)
            if writer is None:
                raise VideoOutputError(
                    "Cannot create MP4 output. Check that OpenCV has video codec support."
                )

            session = _VideoSession(
                output_id=output_id,
                filename=filename,
                path=path,
                writer=writer,
                width=width,
                height=height,
                fps=fps,
                codec=codec,
                updated_at=time.monotonic(),
            )
            self._sessions[output_id] = session
            return self._as_dict(session)

    def _open_writer(
        self, path: Path, fps: float, width: int, height: int
    ) -> tuple[Any | None, str]:
        """Mở VideoWriter với codec tốt nhất khả dụng (M10)."""
        for codec in self.CODEC_PREFERENCE:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = self._writer_factory(str(path), fourcc, fps, (width, height))
            except Exception:  # noqa: BLE001 — codec không có trên build này
                continue
            if writer and writer.isOpened():
                return writer, codec
            if writer:
                writer.release()
        return None, ""

    def append(self, output_id: str, frame: np.ndarray) -> int:
        with self._lock:
            session = self._sessions.get(output_id)
            if session is None:
                raise VideoOutputError("Output session does not exist.")
            if session.complete or session.writer is None:
                raise VideoOutputError("Output session is already complete.")
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                raise VideoOutputError("Annotated frame is invalid.")

            if frame.shape[1] != session.width or frame.shape[0] != session.height:
                frame = cv2.resize(
                    frame,
                    (session.width, session.height),
                    interpolation=cv2.INTER_AREA,
                )
            session.writer.write(np.ascontiguousarray(frame))
            session.frame_count += 1
            session.updated_at = time.monotonic()
            return session.frame_count

    def finish(self, output_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(output_id)
            if session is None:
                raise VideoOutputError("Output session does not exist.")
            if not session.complete:
                if session.writer is not None:
                    session.writer.release()
                session.writer = None
                session.complete = True
                session.updated_at = time.monotonic()

            info = self._as_dict(session)
            info["download_ready"] = session.frame_count > 0 and session.path.is_file()
            info["size_bytes"] = session.path.stat().st_size if session.path.is_file() else 0
            return info

    def get_download(self, output_id: str) -> tuple[Path, str]:
        with self._lock:
            session = self._sessions.get(output_id)
            if session is None or not session.complete:
                raise VideoOutputError("Output is not ready for download.")
            if session.frame_count <= 0 or not session.path.is_file():
                raise VideoOutputError("Output file is empty or missing.")
            return session.path, session.filename

    def _close_stale_sessions(self) -> None:
        """Release abandoned writers before accepting another browser session."""
        now = time.monotonic()
        for session in self._sessions.values():
            if session.complete or now - session.updated_at <= self.STALE_SESSION_SEC:
                continue
            if session.writer is not None:
                session.writer.release()
            session.writer = None
            session.complete = True
            session.updated_at = now

    @staticmethod
    def _as_dict(session: _VideoSession) -> dict:
        return {
            "output_id": session.output_id,
            "filename": session.filename,
            "width": session.width,
            "height": session.height,
            "fps": session.fps,
            "codec": session.codec,
            "frame_count": session.frame_count,
            "complete": session.complete,
        }
