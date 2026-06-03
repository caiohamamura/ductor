"""Handle incoming Telegram media: download, index, and prompt injection."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import ReplyParameters

from ductor_bot.files.image_processor import process_image
from ductor_bot.files.prompt import MediaInfo
from ductor_bot.files.prompt import build_media_prompt as _build_media_prompt_generic
from ductor_bot.files.storage import prepare_destination as _prepare_destination
from ductor_bot.files.storage import sanitize_filename as _sanitize_filename
from ductor_bot.files.storage import update_index
from ductor_bot.messenger.telegram.message_dispatch import ReactionTracker
from ductor_bot.messenger.telegram.topic import get_thread_id

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message

logger = logging.getLogger(__name__)

_TRANSCRIBE_TIMEOUT_SECONDS = 300
_TRANSCRIPT_PREVIEW_LIMIT = 3200


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_media(message: Message) -> bool:
    """True if *message* contains a downloadable media attachment."""
    return bool(
        message.photo
        or message.document
        or message.voice
        or message.video
        or message.audio
        or message.sticker
        or message.video_note
    )


def is_message_addressed(
    message: Message,
    bot_id: int | None,
    bot_username: str | None,
) -> bool:
    """True if *message* in a group chat is addressed to the bot.

    Works for both media (caption entities) and plain text (text entities).
    Checks reply-to-bot, @botname mentions, and /cmd@botname commands.
    """
    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and bot_id is not None
        and message.reply_to_message.from_user.id == bot_id
    ):
        return True

    tag = f"@{bot_username}" if bot_username else None
    for text, entities in (
        (message.caption, message.caption_entities),
        (message.text, message.entities),
    ):
        if not text or not entities or not tag:
            continue
        for e in entities:
            value = text[e.offset : e.offset + e.length].lower()
            if e.type == "mention" and value == tag:
                return True
            if e.type == "bot_command" and value.endswith(tag):
                return True
    return False


def is_command_for_others(
    message: Message,
    bot_username: str | None,
) -> bool:
    """True if *message* is a command explicitly addressed to another bot.

    Checks for /cmd@otherbot patterns in entities/caption_entities.
    """
    if not bot_username:
        return False

    tag = f"@{bot_username.lower()}"
    text = message.text or message.caption or ""
    entities = message.entities or message.caption_entities or []

    for e in entities:
        if e.type == "bot_command":
            cmd = text[e.offset : e.offset + e.length].lower()
            if "@" in cmd and not cmd.endswith(tag):
                return True
    return False


def is_media_addressed(
    message: Message,
    bot_id: int | None,
    bot_username: str | None,
) -> bool:
    """True if a media message in a group chat is addressed to the bot."""
    return is_message_addressed(message, bot_id, bot_username)


def should_drop_in_group(
    message: Message,
    *,
    bot_id: int | None,
    bot_username: str | None,
    group_mention_only: bool,
) -> bool:
    """True if a group/supergroup message should be silently dropped.

    Drops ``/cmd@other_bot`` commands always, and — when *group_mention_only*
    is True — any message not addressed to this bot via reply / @mention /
    ``/cmd@us``. Returns False for non-group chats; the caller can rely on
    this to skip the check in private DMs.
    """
    if message.chat.type not in ("group", "supergroup"):
        return False
    if is_command_for_others(message, bot_username):
        return True
    return group_mention_only and not is_message_addressed(message, bot_id, bot_username)


async def resolve_media_text(
    bot: Bot,
    message: Message,
    telegram_files_dir: Path,
    workspace: Path,
    *,
    status_reaction: bool = True,
) -> str | None:
    """Download media from *message*, update index, return agent prompt.

    Returns ``None`` if the download fails or the message has no media.
    """
    await asyncio.to_thread(telegram_files_dir.mkdir, parents=True, exist_ok=True)

    try:
        info = await download_media(bot, message, telegram_files_dir)
    except (TelegramAPIError, OSError):
        logger.exception("Failed to download media from chat=%d", message.chat.id)
        await message.answer("Could not download that file.")
        return None

    if info is None:
        return None

    try:
        await asyncio.to_thread(update_index, telegram_files_dir)
    except (OSError, yaml.YAMLError):
        logger.warning("Index update failed", exc_info=True)

    if info.original_type in ("voice", "audio"):
        tracker = ReactionTracker(
            bot,
            message.chat.id,
            message.message_id,
            enabled=status_reaction,
        )
        await tracker.set_audio_transcribing()
        status_message = await _send_transcription_status(bot, message)
        transcribed = await _with_audio_transcript(info, workspace)
        info = transcribed if transcribed is not None else _with_transcript_error(info)
        await _edit_transcription_status(bot, status_message, info)

    return build_media_prompt(info, workspace)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_MediaTuple = tuple[str | None, Any, str, str]


async def download_media(bot: Bot, message: Message, base_dir: Path) -> MediaInfo | None:
    """Download the first media attachment into *base_dir*/YYYY-MM-DD/.

    Returns ``None`` when the message contains no supported media.
    """
    kind, file_obj, file_name, mime = _resolve_media(message)
    if kind is None or file_obj is None:
        return None

    dest = await asyncio.to_thread(_prepare_destination, base_dir, file_name)
    await bot.download(file_obj, destination=dest)
    logger.info("Downloaded %s -> %s (%s)", kind, dest, mime)

    dest = await asyncio.to_thread(process_image, dest)
    if dest.suffix == ".webp":
        mime = "image/webp"

    return MediaInfo(
        path=dest,
        media_type=mime,
        file_name=dest.name,
        caption=message.caption,
        original_type=kind,
    )


# ---------------------------------------------------------------------------
# Media extractors
# ---------------------------------------------------------------------------


def _resolve_media(message: Message) -> _MediaTuple:
    """Inspect *message* and return ``(kind, downloadable, filename, mime)``."""
    for extractor in (
        _extract_photo,
        _extract_document,
        _extract_voice,
        _extract_audio,
        _extract_video,
        _extract_video_note,
        _extract_sticker,
    ):
        result = extractor(message)
        if result is not None:
            return result
    return None, None, "", ""


def _extract_photo(msg: Message) -> _MediaTuple | None:
    if not msg.photo:
        return None
    photo = msg.photo[-1]
    return "photo", photo, f"photo_{photo.file_unique_id}.jpg", "image/jpeg"


def _extract_document(msg: Message) -> _MediaTuple | None:
    if not msg.document:
        return None
    doc = msg.document
    name = doc.file_name or f"doc_{doc.file_unique_id}"
    mime = doc.mime_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    return "document", doc, _sanitize_filename(name), mime


def _extract_voice(msg: Message) -> _MediaTuple | None:
    if not msg.voice:
        return None
    v = msg.voice
    return "voice", v, f"voice_{v.file_unique_id}.ogg", v.mime_type or "audio/ogg"


def _extract_audio(msg: Message) -> _MediaTuple | None:
    if not msg.audio:
        return None
    a = msg.audio
    mime = a.mime_type or "audio/mpeg"
    ext = mimetypes.guess_extension(mime) or ".mp3"
    name = a.file_name or f"audio_{a.file_unique_id}{ext}"
    return "audio", a, _sanitize_filename(name), mime


def _extract_video(msg: Message) -> _MediaTuple | None:
    if not msg.video:
        return None
    v = msg.video
    mime = v.mime_type or "video/mp4"
    name = v.file_name or f"video_{v.file_unique_id}.mp4"
    return "video", v, _sanitize_filename(name), mime


def _extract_video_note(msg: Message) -> _MediaTuple | None:
    if not msg.video_note:
        return None
    vn = msg.video_note
    return "video_note", vn, f"videonote_{vn.file_unique_id}.mp4", "video/mp4"


def _extract_sticker(msg: Message) -> _MediaTuple | None:
    if not msg.sticker:
        return None
    s = msg.sticker
    uid = s.file_unique_id
    if s.is_animated:
        return "sticker", s, f"sticker_{uid}.tgs", "application/x-tgsticker"
    if s.is_video:
        return "sticker", s, f"sticker_{uid}.webm", "video/webm"
    return "sticker", s, f"sticker_{uid}.webp", "image/webp"


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


def build_media_prompt(info: MediaInfo, workspace: Path) -> str:
    """Build the Telegram-specific prompt for a received media file."""
    return _build_media_prompt_generic(info, workspace, transport="Telegram")


async def _with_audio_transcript(info: MediaInfo, workspace: Path) -> MediaInfo | None:
    """Return *info* enriched with a transcript, or ``None`` on failure."""
    script = workspace / "tools" / "media_tools" / "transcribe_audio.py"
    if not script.exists():
        logger.warning("Audio transcription tool missing at %s", script)
        return None

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        "--file",
        str(info.path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=_TRANSCRIBE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("Audio transcription timed out for %s", info.path)
        return None

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        logger.warning(
            "Audio transcription failed path=%s rc=%s stderr=%s stdout=%s",
            info.path,
            proc.returncode,
            stderr_text[:500],
            stdout_text[:500],
        )
        return None

    parsed = _parse_transcription_stdout(stdout_text, info.path)
    if parsed is None:
        return None
    transcript, method = parsed
    transcript = transcript.strip()
    if not transcript:
        logger.warning("Audio transcription returned an empty transcript for %s", info.path)
        return None

    return MediaInfo(
        path=info.path,
        media_type=info.media_type,
        file_name=info.file_name,
        caption=info.caption,
        original_type=info.original_type,
        transcript=transcript,
        transcript_method=method,
    )


def _parse_transcription_stdout(stdout_text: str, path: Path) -> tuple[str, str | None] | None:
    try:
        result = json.loads(stdout_text)
    except json.JSONDecodeError:
        return stdout_text, "external"

    if not isinstance(result, dict):
        logger.warning("Audio transcription returned non-object JSON for %s", path)
        return None
    transcript_value = result.get("transcript")
    if not isinstance(transcript_value, str):
        logger.warning("Audio transcription returned no transcript for %s", path)
        return None
    method_value = result.get("method")
    method = method_value if isinstance(method_value, str) else None
    return transcript_value, method


def _with_transcript_error(info: MediaInfo) -> MediaInfo:
    return MediaInfo(
        path=info.path,
        media_type=info.media_type,
        file_name=info.file_name,
        caption=info.caption,
        original_type=info.original_type,
        transcript_error="Direct audio transcription did not produce a transcript.",
    )


async def _send_transcription_status(bot: Bot, message: Message) -> Message | None:
    try:
        return await bot.send_message(
            chat_id=message.chat.id,
            text="\U0001f399\ufe0f Transcrevendo áudio...",
            reply_parameters=ReplyParameters(
                message_id=message.message_id,
                allow_sending_without_reply=True,
            ),
            message_thread_id=get_thread_id(message),
        )
    except TelegramAPIError:
        logger.debug("Failed to send audio transcription status", exc_info=True)
        return None


async def _edit_transcription_status(
    bot: Bot,
    status_message: Message | None,
    info: MediaInfo,
) -> None:
    if status_message is None:
        return

    if info.transcript:
        transcript = html.escape(_truncate_transcript(info.transcript))
        text = f"\u2705 <b>Transcrição do áudio</b>\n\n{transcript}"
    else:
        error = html.escape(info.transcript_error or "Não foi possível transcrever o áudio.")
        text = f"\u26a0\ufe0f <b>Não consegui transcrever o áudio</b>\n\n{error}"

    try:
        await bot.edit_message_text(
            chat_id=status_message.chat.id,
            message_id=status_message.message_id,
            text=text,
            parse_mode=ParseMode.HTML,
        )
    except TelegramAPIError:
        logger.debug("Failed to edit audio transcription status", exc_info=True)


def _truncate_transcript(transcript: str) -> str:
    if len(transcript) <= _TRANSCRIPT_PREVIEW_LIMIT:
        return transcript
    return f"{transcript[:_TRANSCRIPT_PREVIEW_LIMIT].rstrip()}..."
