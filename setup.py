"""Setuptools entrypoint and ArcLM release helper.

Common commands:

    python setup.py --version
    python setup.py release --version 0.1.1
    python setup.py release --version 0.1.1 --readme README.md --upload
    python setup.py release --version 0.1.1 --repository testpypi --upload

    # for new releases: 

        python setup.py release --version 0.1.1 --tag --push
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Command, setup


ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT / "arclm" / "_version.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"
DEFAULT_README = "README.md"
VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')


def read_current_version() -> str:
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(1)


def validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+([a-zA-Z0-9._+-]+)?", version):
        raise ValueError(
            "Version should look like 0.1.0, 0.1.1, 0.2.0, or 1.0.0"
        )


def update_version_file(version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    new_text, count = VERSION_PATTERN.subn(f'__version__ = "{version}"', text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update __version__ in {VERSION_FILE}")
    VERSION_FILE.write_text(new_text, encoding="utf-8")


def update_pyproject_readme(readme: str) -> None:
    readme_path = (ROOT / readme).resolve()
    if not readme_path.exists():
        raise FileNotFoundError(f"Readme file does not exist: {readme}")
    if ROOT not in readme_path.parents and readme_path != ROOT:
        raise ValueError("Readme must be inside the project directory")

    relative_readme = readme_path.relative_to(ROOT).as_posix()
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(?m)^readme\s*=\s*"[^"]+"',
        f'readme = "{relative_readme}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not update readme in {PYPROJECT_FILE}")
    PYPROJECT_FILE.write_text(new_text, encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


class ReleaseCommand(Command):
    """Update version/readme, build distributions, and optionally upload."""

    description = "build and optionally upload a new ArcLM release"
    user_options = [
        ("version=", None, "new package version, for example 0.1.1"),
        ("readme=", None, "readme file to use in pyproject.toml"),
        ("repository=", None, "twine repository name or URL, default: pypi"),
        ("upload", None, "upload dist files with twine"),
        ("tag", None, "create an annotated git tag named vVERSION"),
        ("push", None, "push the current branch and release tag"),
        ("skip-tests", None, "skip pytest before building"),
        ("keep-dist", None, "keep existing files in dist/ instead of cleaning it"),
    ]
    boolean_options = ["upload", "tag", "push", "skip-tests", "keep-dist"]

    def initialize_options(self) -> None:
        self.version = None
        self.readme = DEFAULT_README
        self.repository = "pypi"
        self.upload = False
        self.tag = False
        self.push = False
        self.skip_tests = False
        self.keep_dist = False

    def finalize_options(self) -> None:
        if self.version is None:
            self.version = read_current_version()
        validate_version(self.version)
        if self.push:
            self.tag = True

    def run(self) -> None:
        update_version_file(self.version)
        update_pyproject_readme(self.readme)

        if not self.skip_tests:
            run([sys.executable, "-m", "pytest", "tests"])

        if not self.keep_dist:
            shutil.rmtree(ROOT / "dist", ignore_errors=True)
            shutil.rmtree(ROOT / "build", ignore_errors=True)

        run([sys.executable, "-m", "build"])
        run([sys.executable, "-m", "twine", "check", "dist/*"])

        if self.tag:
            tag_name = f"v{self.version}"
            run(["git", "add", "arclm/_version.py", "pyproject.toml"])
            run(["git", "commit", "-m", f"Release ArcLM {self.version}"])
            run(["git", "tag", "-a", tag_name, "-m", f"ArcLM {self.version}"])

        if self.upload:
            run(
                [
                    sys.executable,
                    "-m",
                    "twine",
                    "upload",
                    "--repository",
                    self.repository,
                    "dist/*",
                ]
            )
        else:
            print("Built and checked dist files. Add --upload to publish them.")

        if self.push:
            run(["git", "push"])
            run(["git", "push", "origin", f"v{self.version}"])


setup(
    name="arclm",
    version=read_current_version(),
    cmdclass={"release": ReleaseCommand},
)
