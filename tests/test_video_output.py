from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.video_output import VideoOutputError, VideoOutputStore


class FakeWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True
        self.path.write_bytes(b"fake-mp4")


def test_video_output_writes_resized_frames_and_finishes(tmp_path):
    created: list[FakeWriter] = []

    def factory(path, _fourcc, _fps, _size):
        writer = FakeWriter(path)
        created.append(writer)
        return writer

    store = VideoOutputStore(tmp_path / "output", writer_factory=factory)
    session = store.start("chuyến đi ../ demo.mp4", width=641, height=481, fps=5)

    assert session["width"] == 640
    assert session["height"] == 480
    assert session["filename"].endswith(".mp4")
    assert "/" not in session["filename"]
    assert "\\" not in session["filename"]

    count = store.append(session["output_id"], np.zeros((240, 320, 3), dtype=np.uint8))
    assert count == 1
    assert created[0].frames[0].shape == (480, 640, 3)

    completed = store.finish(session["output_id"])
    assert completed["complete"] is True
    assert completed["download_ready"] is True
    assert completed["frame_count"] == 1
    assert created[0].released is True

    path, filename = store.get_download(session["output_id"])
    assert path.parent == (tmp_path / "output").resolve()
    assert filename == session["filename"]


@pytest.mark.parametrize(
    ("width", "height", "fps"),
    [(0, 480, 5), (640, 0, 5), (4000, 480, 5), (640, 480, 0)],
)
def test_video_output_rejects_invalid_geometry(tmp_path, width, height, fps):
    store = VideoOutputStore(tmp_path / "output", writer_factory=lambda *args: None)
    with pytest.raises(VideoOutputError):
        store.start("demo.mp4", width=width, height=height, fps=fps)


def test_video_output_rejects_unknown_or_active_download(tmp_path):
    store = VideoOutputStore(
        tmp_path / "output",
        writer_factory=lambda path, *_args: FakeWriter(path),
    )
    session = store.start("demo.mp4", width=640, height=480, fps=5)

    with pytest.raises(VideoOutputError):
        store.get_download(session["output_id"])
    with pytest.raises(VideoOutputError):
        store.append("missing", np.zeros((10, 10, 3), dtype=np.uint8))

    store.finish(session["output_id"])
