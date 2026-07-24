# StatsEngine MCP Server

Exposes StatsEngine's 31 probability/statistics templates as MCP tools, so an
MCP-aware client (Claude Desktop, Claude Code, etc.) can call them directly.
This is an **optional** layer — `core/` and `templates/` never import `mcp`,
and everything here works purely by wrapping the existing
`StatsEngine().execute(...)` and `template_registry` APIs.

## Install

```bash
pip install -r requirements.txt -r requirements-mcp.txt
```

## Tools exposed

- **`list_stats_templates()`** — every registered template's id, description,
  and required params. Call this first.
- **`describe_stats_template(template_id)`** — full parameter schema (types,
  optional params + defaults) for one template.
- **`run_stats_template(template_id, params)`** — runs it. Returns
  `status` (`success` / `needs_input` / `error`), `method`
  (`analytical_math` or `monte_carlo_simulation` — whichever track actually
  produced the result), `value`, and `execution_time_ms`. Missing required
  params come back as `status: "needs_input"` with a `missing` list instead
  of a bare error, so a calling LLM can ask the user and retry.

## Run it

Stdio transport (what local clients like Claude Desktop / Claude Code expect):

```bash
python -m mcp_server.server
```

## Client config

Claude Desktop / Claude Code (`claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "statsengine": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/StatsEngine"
    }
  }
}
```

## Verifying it works standalone

Core StatsEngine has zero dependency on `mcp` — you can prove it by importing
`core`/`templates` in an environment where `mcp` isn't even installed:

```bash
pip uninstall mcp -y   # or just don't install requirements-mcp.txt
python -c "import templates; from core.router import StatsEngine; print('fine')"
```

This should print `fine` — if it doesn't, something leaked an `mcp` import
outside of `mcp_server/`.