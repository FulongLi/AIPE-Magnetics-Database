"""``tomllib`` on Python 3.11+, a minimal fallback parser for the subset used by pyproject.toml on 3.10."""

from __future__ import annotations

try:  # pragma: no cover - depends on interpreter
    from tomllib import loads  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import re

    def loads(text: str) -> dict:  # type: ignore[misc]
        """Tiny TOML subset: tables, string/number/bool scalars, flat arrays of strings. Enough for version checks."""
        data: dict = {}
        current = data
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = data
                for part in line.strip("[]").split("."):
                    current = current.setdefault(part.strip().strip('"'), {})
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip().strip('"'), value.strip()
                if value.startswith('"') and value.endswith('"'):
                    current[key] = value[1:-1]
                elif value in ("true", "false"):
                    current[key] = value == "true"
                elif value.startswith("["):
                    current[key] = re.findall(r'"([^"]*)"', value)
                elif value.startswith("{"):
                    current[key] = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', value))
                else:
                    try:
                        current[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        current[key] = value
        return data
