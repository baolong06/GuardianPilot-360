from __future__ import annotations

from pathlib import Path

import pytest


def _load_server():
    pytest.importorskip("mediapipe")
    import app as server

    return server


class StubVideoOutputs:
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.started_with = None

    def start(self, original_name, width, height, fps):
        self.started_with = (original_name, width, height, fps)
        return {
            "output_id": "safe-session",
            "filename": "demo_analyzed.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": 0,
            "complete": False,
        }

    def finish(self, output_id):
        assert output_id == "safe-session"
        return {
            "output_id": output_id,
            "filename": self.output_file.name,
            "frame_count": 2,
            "complete": True,
            "download_ready": True,
            "size_bytes": self.output_file.stat().st_size,
        }

    def get_download(self, output_id):
        assert output_id == "safe-session"
        return self.output_file, self.output_file.name


def test_video_output_start_finish_and_download(monkeypatch, tmp_path):
    server = _load_server()
    output_file = tmp_path / "demo_analyzed.mp4"
    output_file.write_bytes(b"fake-video")
    store = StubVideoOutputs(output_file)
    monkeypatch.setattr(server, "_initialized", True)
    monkeypatch.setattr(server, "_video_outputs", store)
    client = server.app.test_client()

    started = client.post(
        "/api/video-output/start",
        json={"filename": "demo.mp4", "width": 640, "height": 480, "fps": 5},
    )
    assert started.status_code == 200
    assert started.get_json()["output_id"] == "safe-session"
    assert store.started_with == ("demo.mp4", 640, 480, 5.0)

    finished = client.post("/api/video-output/safe-session/finish")
    assert finished.status_code == 200
    assert finished.get_json()["download_url"].endswith("/download")

    downloaded = client.get("/api/video-output/safe-session/download")
    assert downloaded.status_code == 200
    assert downloaded.data == b"fake-video"
    assert "attachment" in downloaded.headers["Content-Disposition"]


def test_video_analyze_rejects_invalid_source_timestamp(monkeypatch):
    server = _load_server()
    monkeypatch.setattr(server, "_initialized", True)
    client = server.app.test_client()

    response = client.post(
        "/api/analyze",
        json={"image": "not-decoded", "source_timestamp_ms": -1},
    )
    assert response.status_code == 400
    assert "source_timestamp_ms" in response.get_json()["error"]

