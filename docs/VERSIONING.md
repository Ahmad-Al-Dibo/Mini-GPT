# ArcLM Versioning

ArcLM uses two related tools:

- Git commits and branches control source-code history.
- Package versions tell users which ArcLM release they installed.

The package version lives in one place:

```text
arclm/_version.py
```

`pyproject.toml`, `arclm.__version__`, and `arclm.api.get_version()` all read from that file.

## Version Numbers

Use semantic versioning:

```text
MAJOR.MINOR.PATCH
```

- Increase `PATCH` for bug fixes: `0.1.0` -> `0.1.1`
- Increase `MINOR` for new features: `0.1.0` -> `0.2.0`
- Increase `MAJOR` when you break old user code: `1.0.0` -> `2.0.0`

Before the public API is stable, keep using `0.x.y`.

## Current Version

The current ArcLM library version is:

```text
0.1.0
```

You can check it with:

```bash
python -c "import arclm; print(arclm.__version__)"
```

## Release Flow

The easiest release command is:

```bash
python setup.py release --version 0.1.1
```

That command updates `arclm/_version.py`, keeps `pyproject.toml` pointed at
`README.md`, runs tests, builds `dist/`, and checks the files with Twine. It
does not upload unless you ask it to.

Useful examples:

```bash
python setup.py --version
python setup.py release --version 0.1.1 --readme README.md
python setup.py release --version 0.1.1 --repository testpypi --upload
python setup.py release --version 0.1.1 --repository pypi --upload
python setup.py release --version 0.1.1 --tag --push --upload
```

Manual flow:

1. Finish and test the code.
2. Update `arclm/_version.py`.
3. Commit the release.
4. Create a Git tag matching the package version.
5. Push the commit and tag.

Example:

```bash
git status
pytest
git add arclm/_version.py pyproject.toml docs/VERSIONING.md
git commit -m "Release ArcLM 0.1.0"
git tag -a v0.1.0 -m "ArcLM 0.1.0"
git push origin first-pd-version
git push origin v0.1.0
```

The Git tag is how GitHub knows the official release commit.
