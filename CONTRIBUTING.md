# Contributing

## Code style

Lean code and readability come first. Names should reveal intent and never be abbreviations, and we avoid "just in case" code that could lead to drift. See [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md) for the full ground rules.

## Commit format

Every commit message should start with:

```text
[<module>] <type>: <description>
```

`<module>` is the part of the codebase the commit touches, such as `docs`, `tests`, `scripts`, or a top-level package or directory.

`<type>` is one letter:

| Code | Meaning |
|------|---------|
| `B` | Bugfix |
| `W` | Workaround |
| `E` | Enhancement |
| `D` | Documentation |
| `F` | Feature |

## Commit discipline

Write commits that are self-contained and easy to review: one logical change per commit, a header that names the scope and intent, and a body when the *why* is not obvious from the diff. Keep contributions small and targeted.
