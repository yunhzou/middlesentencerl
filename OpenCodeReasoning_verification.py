import argparse, pathlib, sys

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".ruff_cache"}

def check_code(code: str) -> tuple[bool, str]:
    try:
        # compile() is enough; it fully lexes & parses (no execution).
        # Equivalent to ast.parse() but with better error locations.
        compile(code, "<string>", "exec")
        return True, ""
    except (SyntaxError, IndentationError, TabError) as e:
        # Build a concise, grep-friendly error message
        loc = f"<string>:{getattr(e, 'lineno', '?')}:{getattr(e, 'offset', '?')}"
        line = (e.text or "").rstrip("\n")
        caret = " " * (max((e.offset or 1) - 1, 0)) + "^" if e.offset else ""
        msg = f"{loc}: {e.__class__.__name__}: {e.msg}\n    {line}\n    {caret}"
        return False, msg
