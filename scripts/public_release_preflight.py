#!/usr/bin/env python3
"""Fail if the curated Routide public artifact crosses its release boundary."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_TOP_LEVEL = {
    ".gitattributes",
    ".github",
    ".gitignore",
    ".swift-format",
    "App",
    "CITATION.cff",
    "LICENSE",
    "NOTICE.md",
    "Package.swift",
    "README.md",
    "Research",
    "Sources",
    "Tests",
    "artifacts",
    "docs",
    "paper",
    "scripts",
}

FORBIDDEN_PATH_PARTS = (
    ".tfignore",
    "support/git_snapshot.py",
    "Research/Paper/Submission",
    "Research/DeviceRoutes",
    "Research/DeviceRuns",
)

FORBIDDEN_SUFFIXES = (
    ".p12",
    ".mobileprovision",
    ".safetensors",
    ".gguf",
    ".dtps",
    ".trace",
    ".ips",
    ".zip",
)

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".jsonl",
    ".md",
    ".plist",
    ".py",
    ".swift",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_PATTERNS = {
    "private Azure DevOps URL": re.compile(r"dev\.azure\.com", re.I),
    "private Azure organization": re.compile(r"bruhmoment123", re.I),
    "historical local username": re.compile(r"blue9628", re.I),
    "historical repository name": re.compile(r"Routide-History", re.I),
    "TFVC migration marker": re.compile(r"\bTFVC\b", re.I),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\Users\\\\", re.I),
    "macOS user path": re.compile(r"/Users/[^/\s]+/", re.I),
    "OneDrive path": re.compile(r"\bOneDrive\b", re.I),
    "author email": re.compile(r"musa\.shams@", re.I),
    "GitHub classic token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

REQUIRED_PATHS = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE.md",
    "Package.swift",
    "paper/routide-arxiv.tex",
    "paper/routide-arxiv.pdf",
    "artifacts/public_results/results.json",
    "scripts/verify_public_results.py",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    top = {path.name for path in ROOT.iterdir() if path.name != ".git"}
    unexpected = sorted(top - ALLOWED_TOP_LEVEL)
    if unexpected:
        fail(f"unexpected top-level paths: {unexpected}")

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            fail(f"missing required public artifact: {relative}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    public_url = "https://github.com/MusaShams/Routide"
    if public_url not in readme or public_url not in citation:
        fail("public repository URL is missing from README/CITATION metadata")

    scanned = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()

        if path.stat().st_size >= 10 * 1024 * 1024:
            fail(f"public file is unexpectedly large: {relative}")

        if any(part.lower() in relative.lower() for part in FORBIDDEN_PATH_PARTS):
            fail(f"forbidden public path: {relative}")

        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"forbidden public file type: {relative}")

        suffix = path.suffix.lower()
        if path.name in {".gitignore", ".gitattributes"}:
            suffix = path.name
        if suffix not in TEXT_SUFFIXES:
            continue
        if relative == "scripts/public_release_preflight.py":
            # This file necessarily contains the forbidden-pattern definitions.
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        scanned += 1
        for label, pattern in CONTENT_PATTERNS.items():
            if pattern.search(content):
                fail(f"{label} found in public file: {relative}")

        if "Anonymous Author(s)" in content:
            fail(f"anonymous-review identity marker found in public file: {relative}")

    print(f"Routide public-release preflight: OK ({scanned} text files scanned)")


if __name__ == "__main__":
    main()
