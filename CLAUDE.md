# CLAUDE.md

## Git exception

For the /work-issue workflow, `git commit` and `git push` are allowed only on `issue-<N>-*` branches the run creates; never commit or push to `main`. PRs are merged by the orchestrator (squash) after checks pass, never by the run itself. Never use emojis or em dashes in files; never write TypeScript.
