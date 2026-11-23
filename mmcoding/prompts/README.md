# Prompt texts

Everything the policy model reads besides the task instruction, loaded by
`mmcoding/host/deploy.py` (`rv_runtime_prompt`). One trailing newline of each file is ignored;
every other character, including trailing spaces, is part of the prompt.

The prompt of every harness is

```
You are operating the task's Ubuntu desktop. <capability/<harness>.txt>
<closing.txt>
<report_note.txt>
<task_date.txt>          (tasks with a reference date)
<ask_user.txt>           (OSWorld 2.0 tasks with a user simulator)

TASK INSTRUCTION (authoritative, verbatim):
<instruction>
```

`capability/<harness>.txt` describes the harness's tools; `{with_x}` and `{serve_note}` are
replaced by `with_x.txt` (display access of sandboxed code) and `serve_note.txt` (`rv-serve`).
The harnesses differ only in this paragraph.
