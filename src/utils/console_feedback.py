import sys
import threading

_CONSOLE_LOCK = threading.Lock()
_PROGRESS_ACTIVE = False
_PROGRESS_LINE_LENGTH = 0


def _clear_progress_line_locked() -> None:
    global _PROGRESS_ACTIVE, _PROGRESS_LINE_LENGTH

    if not _PROGRESS_ACTIVE:
        return

    sys.stdout.write("\r" + (" " * _PROGRESS_LINE_LENGTH) + "\r")
    sys.stdout.flush()
    _PROGRESS_ACTIVE = False
    _PROGRESS_LINE_LENGTH = 0


def console_feedback(message: str) -> None:
    """Print a short user-facing progress line immediately."""
    with _CONSOLE_LOCK:
        _clear_progress_line_locked()
        print(f"✓ {message}", flush=True)


def console_progress(
    label: str,
    *,
    total: int,
    completed: int,
    uploaded: int,
    skipped: int,
    failed: int,
    active: int = 0,
) -> None:
    """Render a compact single-line progress bar without spamming the console."""
    global _PROGRESS_ACTIVE, _PROGRESS_LINE_LENGTH

    total = max(0, int(total))
    completed = max(0, min(int(completed), total))
    uploaded = max(0, int(uploaded))
    skipped = max(0, int(skipped))
    failed = max(0, int(failed))
    active = max(0, int(active))

    width = 24
    filled = width if total == 0 else int(round((completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    line = (
        f"↻ {label} [{bar}] {completed}/{total} | "
        f"uploaded={uploaded} | skipped={skipped} | failed={failed}"
    )
    if active > 0:
        line = f"{line} | threads={active}"

    with _CONSOLE_LOCK:
        padded = line.ljust(max(_PROGRESS_LINE_LENGTH, len(line)))
        sys.stdout.write("\r" + padded)
        if completed >= total:
            sys.stdout.write("\n")
            _PROGRESS_ACTIVE = False
            _PROGRESS_LINE_LENGTH = 0
        else:
            _PROGRESS_ACTIVE = True
            _PROGRESS_LINE_LENGTH = len(padded)
        sys.stdout.flush()
