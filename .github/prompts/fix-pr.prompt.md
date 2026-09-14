---
name: Fix PR CI and Review Comments
description: "Use when: triaging failing GitHub Actions checks and reviewer feedback on a pull request or branch; separate actionable failures from infrastructure flakes, judge each review comment on user benefit versus maintenance and complexity cost, fix and validate locally, then commit and push."
argument-hint: "Optional PR number, URL, or branch; otherwise use the active pull request / current branch"
agent: "agent"
tools: [read, search, edit, execute, todo, agent, web, mcp_github/*]
---
Triage and fix CI failures, review feedback, and supplied Open3D issues as one
work set. Follow `AGENTS.md` and `AGENTS.local.md`; the repository and current
worktree are authoritative. This prompt adds PR-specific triage rules.

## Progress Report

Before editing, create or update the required Markdown progress report. Lock
the goal, requirements, test method, risks, and implementation status. Include
a root-cause table and keep it current:

| Item | Type | Likely root cause / group | Verdict | Reproducer or evidence | Resolution | Validation | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |

Use one row per CI failure, issue, or comment. Group items sharing a likely
cause under the same concise group name. Record unknown causes as
`investigating`; revise the table when evidence disproves a grouping.

## Workflow

1. **Identify.** Resolve the PR from the argument, else the active PR / current
   branch. Stop and ask if there is no PR, several candidates, a detached HEAD,
   or a fork head you cannot push to.

2. **Collect.** Read each supplied issue and comment, including its discussion
   and linked PRs. For descriptions, search for duplicates. Read the checks for
   the PR head commit and the logs of each failing job. Flag results from an
   older commit as stale, and unpushed local commits as not yet covered by CI.
   Read review comments and requested changes, skipping outdated or resolved
   threads. Add every CI failure, issue, and review comment to the report table.

3. **Triage and group.** Classify each failure as actionable (code, test,
   build, style, docs, or config defect), infrastructure (download, runner,
   timeout, quota, cancelled), or flaky/unrelated (also fails on base, or
   untouched by the diff). Only fix the first; report the rest without claiming
   they were fixed. Re-run a check only when a transient cause is likely and
   re-running is cheap. Capture the cheapest decisive evidence and assign every
   issue or comment a verdict: `accept`, `needs clarification`, `duplicate`,
   `cannot reproduce`, `expected behavior`, or `out of scope`. Prioritize and
   investigate by root-cause group; do not implement rejected items.

4. **Classify each review comment** as valid or non-actionable, with a one-line
   reason weighing user benefit against maintenance and complexity cost.
   Non-actionable: subjective, obsolete, unsupported by the code, or outside the
   PR's scope. Ask instead of guessing when a comment implies an API, product,
   or scope decision.

5. **Fix and validate.** Reproduce locally first where possible, and follow the
   debugging process in `AGENTS.md`: try the easy candidate fixes directly, and
   instrument before attempting a hard or complex one. Implement the smallest
   fix that resolves every accepted item in a group without masking distinct
   causes. Keep individual report rows current with the fix or explicit non-fix
   resolution. Stay inside the PR's existing scope; do not opportunistically
   refactor. Name the configurations (CUDA, SYCL, macOS, Windows) you could not
   verify locally.

6. **Iterate and deliver.** Continue until every actionable item is fixed or a
   concrete blocker remains. Inspect the branch, active PR, remotes, and
   worktree before git writes. Continue on a user-selected branch or active
   related PR; otherwise ask before choosing or switching branches. Commit only
   the fixes, exclude the progress report, reference every resolved issue number
   in the commit or PR, then push. Do not open a duplicate PR for an existing
   one.

## Validation

`complete` validation is a relevant test built from the changed code that
failed before the change and passes afterward. Record the failing baseline and
passing command or result in the report. Linting, static checks, and `git diff
--check` are never validation evidence.

When complete validation is not possible, perform the strongest available
check and mark the affected item or group `partial` (for example, it builds but
is not run) or `absent` (the code was not built). State the limitation and
reason in the report and final summary; never describe partial or absent
validation as complete.

## Stop Conditions

Leave an item unresolved, with its reason in the report table, when:

- its verdict does not support a fix;
- resolution needs a product or API decision; or
- required access, hardware, or a clean branch transition is unavailable.

Do not merge or close the PR, and do not post comments, reviews, or thread
resolutions unless the user explicitly asks. Do not create, close, label, or
comment on issues unless the user explicitly requests that GitHub mutation.

## Final Report

- actionable CI failures and their root causes;
- each review comment, marked valid or non-actionable, with the reason;
- fixes made, with file links;
- validation commands actually run, their results, and unverified platforms;
- ignored infrastructure/flaky failures and remaining blockers;
- completed and unresolved items by root-cause group, with their `complete`,
  `partial`, or `absent` validation status and evidence;
- branch, commit hash, push result, PR, risks, and the CI run to watch.