# Codex and Claude Code plugins

Both plugins are thin: they ship the same five skills and the same `.mcp.json` that starts the
local `aipe-magnetics` MCP server. **All engineering numbers come from the Python engine**; the
plugins contain no calculations. Standalone installation needs only Python 3.10+ and the
`aipe-magnetics` package - no AIPE Orchestrator, no Registry service, no other AIPE domain expert,
no commercial software and no LLM API credentials (the hosts bring their own model access).

| | Claude Code | OpenAI Codex |
| --- | --- | --- |
| plugin root | `plugins/claude/aipe-magnetics/` | `plugins/codex/aipe-magnetics/` |
| manifest | `.claude-plugin/plugin.json` | `.codex-plugin/plugin.json` |
| marketplace (repository root) | `.claude-plugin/marketplace.json` | `.agents/plugins/marketplace.json` |
| skills | `skills/<name>/SKILL.md` (×5) | `skills/<name>/SKILL.md` (×5, identical) |
| MCP server | `.mcp.json` → `python -m aipe_magnetics.mcp.server` | `.mcp.json` → same command |
| validated with | `claude plugin validate` (Claude Code 2.1.80; newer versions add `--strict`) | `codex plugin marketplace add` + `codex plugin add` (codex-cli 0.163.0) |

Skills: `magnetic-material-analysis`, `inductor-design`, `transformer-design`,
`magnetic-loss-modelling`, `magnetic-engineering-review`. The canonical sources live in `skills/`;
`python tools/sync_plugin_skills.py --check` guarantees both plugin copies are identical.

## 1. Install the engine (both hosts)

```sh
python -m pip install "aipe-magnetics[mcp] @ git+https://github.com/FulongLi/AIPE-Magnetics-Database@feature/aipe-magnetics-v0.2"
# or from a checkout:
python -m pip install -e ".[mcp]"
python -m aipe_magnetics.mcp.server --self-test   # prints the tool list and exits 0
```

`.mcp.json` launches `python -m aipe_magnetics.mcp.server`. The `python` on the host's PATH must
be the interpreter where the package is installed. If you use a virtual environment, either start
the host from the activated environment or edit the installed plugin's `.mcp.json` command to the
absolute interpreter path (e.g. `C:\\venvs\\aipe\\Scripts\\python.exe` or `/opt/aipe/bin/python`).

## 2. Claude Code

```text
/plugin marketplace add FulongLi/AIPE-Magnetics-Database        # GitHub (default branch) ...
/plugin marketplace add /path/to/AIPE-Magnetics-Database         # ... or a local checkout
/plugin install aipe-magnetics@aipe-magnetics                    # opens the details panel; choose a scope
```

Shell equivalents: `claude plugin marketplace add <source>`, `claude plugin install
aipe-magnetics@aipe-magnetics --scope user|project|local`. For a one-session test without a
marketplace: `claude --plugin-dir plugins/claude/aipe-magnetics`.

Use: `/aipe-magnetics:transformer-design`, `/aipe-magnetics:magnetic-material-analysis`, ... or just
ask ("compare N87 and 3C94 at 100 kHz, 150 mT, 90 degC"); Claude invokes the matching skill and the
MCP tools `list_materials`, `compare_materials`, `estimate_core_loss`, `design_transformer`, ...

Uninstall / disable:

```sh
claude plugin disable aipe-magnetics
claude plugin uninstall aipe-magnetics            # add --scope project|local if installed there
claude plugin marketplace remove aipe-magnetics
```

Cache locations: `~/.claude/plugins/cache/aipe-magnetics/aipe-magnetics/<version>/`
(`${CLAUDE_PLUGIN_ROOT}`), marketplaces under `~/.claude/plugins/marketplaces/`.

## 3. OpenAI Codex

Plugins are supported in Codex CLI (`/plugins` browser) and Codex in the ChatGPT desktop app; the
IDE extension does not load plugins but does load standalone skills (see below).

```sh
codex plugin marketplace add https://github.com/FulongLi/AIPE-Magnetics-Database   # or a local path
codex plugin add aipe-magnetics@aipe-magnetics
codex mcp list                      # shows aipe-magnetics (launched from the plugin)
```

Start a new Codex session; invoke skills with `$transformer-design`, `$magnetic-loss-modelling`,
... or let Codex pick them from their descriptions. `config.toml` gains
`[marketplaces.aipe-magnetics]` and `[plugins."aipe-magnetics@aipe-magnetics"] enabled = true`; the
plugin is copied to `~/.codex/plugins/cache/aipe-magnetics/aipe-magnetics/<version>/`.

Uninstall / disable:

```sh
codex plugin remove aipe-magnetics@aipe-magnetics
codex plugin marketplace remove aipe-magnetics
# or keep it installed but disabled: [plugins."aipe-magnetics@aipe-magnetics"] enabled = false
```

Standalone skills (IDE extension, or without the plugin system): copy `skills/*` into
`~/.agents/skills/` (user) or `<repo>/.agents/skills/` (project) and register the MCP server
manually:

```sh
codex mcp add aipe-magnetics -- python -m aipe_magnetics.mcp.server
```

## 4. Troubleshooting

| Symptom | Check |
| --- | --- |
| MCP server does not start / tools missing | `python -m aipe_magnetics.mcp.server --self-test` in a terminal. `ModuleNotFoundError: mcp` → `pip install "aipe-magnetics[mcp]"`. Wrong interpreter → set the absolute path in `.mcp.json`. |
| Host says the plugin failed to load | `claude plugin validate plugins/claude/aipe-magnetics` (add `--strict` on Claude Code ≥ 2.1.233); `codex doctor`. The manifest must stay inside `.claude-plugin/` / `.codex-plugin/`; skills and `.mcp.json` at the plugin root. |
| Skill does not trigger | Invoke it explicitly (`/aipe-magnetics:<skill>` or `$<skill>`); start a new session after installing. |
| `out_of_range` or `rejected` results | Not a plugin problem: the engine refuses to extrapolate outside the material parameter-set ranges. Read the `errors`/`limitations` fields. |
| Writing a report file is refused | By design: start the server with `AIPE_MAGNETICS_MCP_ALLOW_WRITE=1` (or `--allow-write --output-dir <dir>`) and pass an explicit `output_path` inside that directory. |
| Windows path issues | Use forward slashes or escaped backslashes in `.mcp.json`; the server itself uses `pathlib`. |

## 5. Security notes

The MCP server uses stdio only, never opens a network port, never executes repository content or
shell commands, and writes files only when explicitly authorised with an output directory. No API
keys are read or required.
