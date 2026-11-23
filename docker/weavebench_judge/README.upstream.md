# WeaveBench Agent-as-Judge Template

This tarball is unpacked by `weavebench-download-judge` to provide the
host-side OpenClaw profile + workspace that the trajectory-aware judge
expects.

Layout after `tar xzf` into `~/judge_agent_test/`:

```
~/judge_agent_test/
├── template_profile/      <- this template
│   └── openclaw.json      <- placeholder OpenRouter config (filled in by setup)
└── template_workspace/    <- this template (openclaw starter pack)
    ├── AGENTS.md
    ├── IDENTITY.md
    └── ...
```

`scripts/setup.sh` (or `weavebench-download-judge`) substitutes
`PLACEHOLDER_OPENROUTER_API_KEY` with `$OPENROUTER_API_KEY` and
`PLACEHOLDER_WORKSPACE_DIR` with the absolute judge workspace path
before the first `weavebench-run` invocation.

`PLACEHOLDER_GATEWAY_TOKEN` is regenerated to a random hex string per
install — it's only used for openclaw's local loopback gateway and
never leaves the host.
