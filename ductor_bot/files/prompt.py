"""Transport-agnostic media prompt building."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Metadata for a received media file (from any transport)."""

    caption: str | None
    file_name: str
    media_type: str
    original_type: str
    path: Path
    transcript: str | None = None
    transcript_method: str | None = None
    transcript_error: str | None = None


def build_media_prompt(
    info: MediaInfo,
    workspace: Path,
    *,
    transport: str = "",
) -> str:
    """Build the prompt injected into the orchestrator for a received file.

    Paths are relative to *workspace* so they work in both host and Docker.
    """
    rel_path: Path | str = info.path
    with contextlib.suppress(ValueError):
        rel_path = info.path.relative_to(workspace)

    via = f" via {transport}" if transport else ""
    lines = [
        "[INCOMING FILE]",
        f"The user sent you a file{via}.",
        f"Path: {rel_path}",
        f"Type: {info.media_type}",
        f"Original filename: {info.file_name}",
        "",
        "Check tools/media_tools/CLAUDE.md for file handling instructions.",
    ]

    if info.original_type in ("voice", "audio") and info.transcript:
        lines.extend(
            [
                "",
                "[VOICE MESSAGE TRANSCRIPT]",
                info.transcript,
            ]
        )
        if info.transcript_method:
            lines.append(f"Transcription method: {info.transcript_method}")
        lines.append("Respond to the user's transcribed voice message.")
    elif info.original_type in ("voice", "audio"):
        lines.extend(
            [
                "",
                "[VOICE MESSAGE TRANSCRIPTION FAILED]",
                info.transcript_error or "No transcript was produced.",
                "Tell the user the voice message could not be transcribed and ask them to resend or clarify.",
            ]
        )

    if info.original_type in ("video", "video_note"):
        lines.append(
            "This is a video file. Use "
            f"tools/media_tools/process_video.py --file {rel_path} "
            "to extract keyframes and transcribe audio, then respond to the content."
        )

    if info.caption:
        lines.append("")
        lines.append(f"User message: {info.caption}")

    return "\n".join(lines)
