"""Inspect distribution members for local files and embedded credentials."""

import argparse
import re
import tarfile
import zipfile
from pathlib import PurePosixPath

PRIVATE_PARTS = {
    ".git",
    ".claude",
    "__pycache__",
}
PRIVATE_SUFFIXES = {".ipynb", ".ods", ".xlsx", ".pyc", ".log"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".cff", ".txt", ".yml", ".yaml", ".in"}
PATTERNS = {
    "personal absolute path": re.compile(r"/(?:home|Users)/[^\s'\"/]+/"),
    "cloud access key": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "GitHub credential": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}


def members(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    yield entry.filename, archive.read(entry)
    else:
        with tarfile.open(path, "r:*") as archive:
            for entry in archive.getmembers():
                if entry.issym() or entry.islnk():
                    raise ValueError("Distribution contains a link: " + entry.name)
                if entry.isfile():
                    with archive.extractfile(entry) as stream:
                        yield entry.name, stream.read()


def check(path):
    failures, count = [], 0
    for name, data in members(path):
        count += 1
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            failures.append((name, "unsafe archive path"))
        if PRIVATE_PARTS.intersection(member.parts) or member.suffix in PRIVATE_SUFFIXES:
            failures.append((name, "local analysis or development file"))
        if member.suffix in TEXT_SUFFIXES or member.name in {"PKG-INFO", "METADATA", "README", "LICENSE"}:
            text = data.decode("utf-8", errors="replace")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    failures.append((name, label))
    for name, label in failures:
        print(f"{path}: {name}: {label}")
    if failures:
        raise ValueError(f"{path}: {len(failures)} distribution-content failures")
    print(f"{path}: {count} files inspected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+")
    for path in parser.parse_args().archives:
        check(path)


if __name__ == "__main__":
    main()
