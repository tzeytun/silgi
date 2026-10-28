# Contributing to silgi

Thanks for helping! silgi is a small project with a few firm rules (see
[Principles](#principles)); everything else is up for discussion. If you are
unsure whether something fits, open an issue first.

## Principles

1. **No AI/ML.** Classical image processing only: NumPy, OpenCV, SciPy, Numba.
   No neural networks, pretrained models or model downloads.
2. **Offline.** Nothing may make a network request at runtime.
3. **Non-destructive.** Pixels outside the mask must stay bit-identical. Every
   removal method goes through `core/roi.py::run_in_roi`, which enforces this.
4. **Quality preserved.** Never silently change resolution, bit depth, alpha or
   ICC profile.
5. **Deterministic.** Same input, mask and settings give the same bytes. Seed
   anything random.

## Development setup

You need Python 3.11+ and Node 20+.

```bash
git clone https://github.com/tzeytun/silgi && cd silgi
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install

cd frontend && npm install && cd ..
```

Run the app while developing:

```bash
silgi --no-browser --port 8765        # API + built frontend
cd frontend && npm run dev            # Vite dev server on :5173, proxies /api to :8765
```

Build the frontend into the Python package:

```bash
cd frontend && npm run build          # writes to src/silgi/web/
```

## Running the tests

```bash
pytest                                # Python suite (outbound network is blocked)
pytest --cov=silgi.core               # with coverage (core must stay at 85 % or more)
ruff check . && ruff format --check .
mypy                                  # strict for silgi.core

cd frontend
npm run lint && npm run format:check
npm test                              # Vitest
npx playwright install chromium       # once
npm run test:e2e                      # end-to-end, starts the real server
```

### Quality baselines

`tests/baselines/quality.json` records PSNR/SSIM for every method on synthetic
scenes. The test fails if quality drops by more than 0.5 dB / 0.005 SSIM. After an
intentional improvement, refresh the baselines and include the diff in your PR:

```bash
SILGI_UPDATE_BASELINES=1 pytest tests/core/test_quality.py
```

## Code style

* Python: `ruff` (lint and format, line length 100) and `mypy --strict` on
  `silgi.core`. Type everything in `core`.
* TypeScript: ESLint and Prettier (`npm run lint`, `npm run format`).
* Match the surrounding code: naming, comment density and idiom. Explain *why*,
  not *what*.
* UI strings go through i18n: add them to **both** `frontend/src/i18n/en.ts` and
  `tr.ts` (a test fails if the keys or placeholders differ).
* Sample and test images must be generated in code (`src/silgi/samples.py`). Do not
  add third-party or copyrighted images.

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`.
Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
Keep commits small and meaningful; explain the reason in the body when it is not
obvious.

```
fix(patchmatch): keep source patches inside the user's source mask
feat(select): add polygon lasso
```

## Adding a removal method

1. **Write the module** `src/silgi/core/mymethod.py` with
   ```python
   def fill(roi: FloatImage, mask: BoolMask, **params) -> FloatImage: ...
   def remove(image: Image, mask: MaskLike, **params) -> Image:
       return run_in_roi(image, mask, lambda r, m, box: fill(r, m, **params), margin=...)
   ```
   `fill` receives a float32 `HxWxC` crop in `[0, 1]`, the crop's boolean mask and
   its box, and returns an array of the same shape. Never write outside the mask
   yourself; `run_in_roi` does the paste-back. Report progress and honour
   cancellation through `Progress` if the method is slow.
2. **Register it** in `core/engine.py`: add it to `METHODS`, give it a one-line
   plain-English entry in `DESCRIPTIONS`, and dispatch to it in `remove`.
3. **Expose it**: add the name to `MethodName` in `api/schemas.py` and the CLI
   `METHODS`; in the frontend add it to `METHODS` in `RightPanel.tsx` and add its
   name and help text to `en.ts` and `tr.ts`.
4. **Test it**
   * add it to the parametrised untouched-pixels test in
     `tests/core/test_untouched.py`,
   * add it to the determinism test in `tests/core/test_methods.py`,
   * add it to `METHODS` in `tests/quality_cases.py` and record baselines.
5. **Document it** in `docs/algorithms.md` with plain-language explanation and
   references, and add a line to `CHANGELOG.md`.

## Pull requests

* One focused change per PR; describe what and why.
* All checks must pass (the CI runs Linux, macOS and Windows on Python 3.11–3.13).
* Add or update tests, and docs for user-visible changes.

By contributing you agree that your contribution is licensed under the MIT
License, as the rest of the project.
