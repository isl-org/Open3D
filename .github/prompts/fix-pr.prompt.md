---
name: Fix PR CI and Review Comments
description: "Use when: triaging failing GitHub Actions checks and reviewer feedback on a pull request or branch; separate actionable failures from infrastructure flakes, judge each review comment on user benefit versus maintenance and complexity cost, fix and validate locally, then commit and push."
argument-hint: "Optional PR number, URL, or branch; otherwise use the active pull request / current branch"
agent: "agent"
tools: [read, search, edit, execute, todo, agent, web, mcp_github/*]
---
Triage and fix CI failures and review feedback for a pull request. Follow
`AGENTS.md` and `AGENTS.local.md`; this prompt only adds PR-specific triage
rules.

## Workflow

1. **Identify.** Resolve the PR from the argument, else the active PR / current
   branch. Stop and ask if there is no PR, several candidates, a detached HEAD,
   or a fork head you cannot push to.

2. **Collect.** Read the checks for the PR head commit and the logs of each
   failing job. Flag results from an older commit as stale, and unpushed local
   commits as not yet covered by CI. Read review comments and requested changes,
   skipping outdated or resolved threads.

3. **Classify each failure** as actionable (code, test, build, style, docs, or
   config defect), infrastructure (download, runner, timeout, quota, cancelled),
   or flaky/unrelated (also fails on base, or untouched by the diff). Only fix
   the first; report the rest without claiming they were fixed. Re-run a check
   only when a transient cause is likely and re-running is cheap.

4. **Classify each review comment** as valid or non-actionable, with a one-line
   reason weighing user benefit against maintenance and complexity cost.
   Non-actionable: subjective, obsolete, unsupported by the code, or outside the
   PR's scope. Ask instead of guessing when a comment implies an API, product,
   or scope decision.

5. **Fix and validate.** Reproduce locally first where possible, and follow the
   debugging process in `AGENTS.md`: try the easy candidate fixes directly, and
   instrument before attempting a hard or complex one. Stay inside the PR's
   existing scope; do not opportunistically refactor. Name the configurations
   (CUDA, SYCL, macOS, Windows) you could not verify locally.

6. **Iterate** until every actionable item is fixed or a concrete blocker
   remains, then commit and push.

Do not merge or close the PR, and do not post comments, reviews, or thread
resolutions unless the user explicitly asks.

## Final Report

- actionable CI failures and their root causes;
- each review comment, marked valid or non-actionable, with the reason;
- fixes made, with file links;
- validation commands actually run, their results, and unverified platforms;
- ignored infrastructure/flaky failures and remaining blockers;
- commit hash, push result, and the CI run to watch.