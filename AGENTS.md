# AGENTS.md

Instructions for AI agents (Claude Code, Copilot, Codex) working in this repository.

## Required Workflow

For any non-trivial change, follow this sequence:

1. Read the relevant code and existing docs first.
2. Create a new spec in `docs/specs/` or update the existing spec that already owns that behavior.
3. Implement only after the spec exists.
4. Update durable docs if shipped behavior changed.
5. Verify the implementation with the most relevant tests or validation commands available.

Default to a spec for:

- feature work
- CLI behavior changes
- install/uninstall UX changes
- provider/parser changes
- masking and restoration behavior
- pattern/schema changes
- dashboard behavior
- test strategy changes

For very small changes, keep the ceremony small:

- typo-only changes may skip a new spec
- tightly scoped bug fixes should still update an existing spec or add a short new one

## Spec Rules

- Prefer updating an existing spec when the behavior already has a home in `docs/specs/`.
- Create a new spec when the change introduces a distinct behavior, workflow, or subsystem concern.
- Keep specs concrete. Describe user-visible behavior, config shape, edge cases, and acceptance criteria.
- Specs should reflect intended behavior before implementation, not just summarize the code after the fact.

Use `docs/specs/spec-template.md` when creating a new spec.

## Implementation Follow-Through

After code changes, check whether these also need updates:

- `README.md`
- `CLAUDE.md`
- `docs/index.md`
- `docs/testing-scenarios.md`
- `CHANGELOG.md`

If user-visible behavior changed and the docs were not updated, the work is incomplete.

## Verification Expectations

Before closing work:

- run the narrowest useful tests first
- run broader validation if the change crosses subsystem boundaries
- call out anything you could not verify

Avoid claiming behavior that is not reflected in code, tests, or current docs.
