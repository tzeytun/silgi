## What and why

<!-- What does this change and why is it needed? Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran, images you tried, screenshots for UI changes. -->

## Checklist

- [ ] The commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] `pytest`, `ruff check .`, `ruff format --check .` and `mypy` pass
- [ ] Frontend changes: `npm run lint`, `npm test` and `npm run build` pass
- [ ] Tests added or updated (new removal methods are in the untouched-pixels and determinism tests)
- [ ] Pixels outside the mask are still bit-identical; no network access added; no AI/ML dependency added
- [ ] User-visible strings exist in both English and Turkish
- [ ] Docs and `CHANGELOG.md` updated where relevant
