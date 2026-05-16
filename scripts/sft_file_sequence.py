# raw_generations 내부의 순번 파일명을 계산한다.
# request_0001.json, batch_0001_raw.jsonl 같은 파일명을 만든다.
# 기존 파일을 덮어쓰지 않도록 현재 최대 번호 다음 번호를 사용한다.

from __future__ import annotations

import re
from pathlib import Path


def existing_numbers(directory: Path, prefix: str, suffix: str) -> list[int]:
    if not directory.exists():
        return []

    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d{{4}}){re.escape(suffix)}$"
    )

    numbers: list[int] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match:
            numbers.append(int(match.group(1)))

    return sorted(numbers)


def next_numbered_index(directory: Path, prefix: str, suffix: str) -> int:
    numbers = existing_numbers(directory, prefix, suffix)
    return 1 if not numbers else numbers[-1] + 1


def numbered_path(directory: Path, prefix: str, number: int, suffix: str) -> Path:
    return directory / f"{prefix}_{number:04d}{suffix}"


def next_numbered_path(directory: Path, prefix: str, suffix: str) -> Path:
    number = next_numbered_index(directory, prefix, suffix)
    return numbered_path(directory, prefix, number, suffix)