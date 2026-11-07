"""Build hook: bundle the committed data store (data/normalized, data/canonical, data/sources.json, data/reports, data/huggingface.json) into the
wheel as ``aipe_magnetics_database/_bundled`` so that an installed package works without a checkout. Regenerated
local-only table sidecars (``tables-local/``) are never bundled."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
BUNDLED = ("normalized", "canonical", "reports", "sources.json", "huggingface.json")


class BuildWithData(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "aipe_magnetics_database" / "_bundled"
        if target.exists():
            shutil.rmtree(target)
        for name in BUNDLED:
            source = ROOT / "data" / name
            if source.is_dir():
                shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("tables-local", "*.tmp"))
            elif source.is_file():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target / name)


setup(cmdclass={"build_py": BuildWithData})
