from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return workspace_root() / "data"


def _candidate_roots() -> list[Path]:
    root = workspace_root()
    candidates = [data_root(), root]
    parent = root.parent
    if parent != root:
        candidates.append(parent)
    return candidates


def resolve_existing_path(*parts: str) -> Path:
    for root in _candidate_roots():
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return workspace_root().joinpath(*parts)


def resolve_existing_aliases(options: list[tuple[str, ...]]) -> Path:
    for option in options:
        for root in _candidate_roots():
            candidate = root.joinpath(*option)
            if candidate.exists():
                return candidate
    return workspace_root().joinpath(*options[0])


def resolve_user_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for root in _candidate_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    return workspace_root() / path


def kinface_workspace_root() -> Path:
    return resolve_existing_aliases(
        [
            ("kinface",),
            ("Kinship Verification project_ Source Code",),
        ]
    )


def kinver_workspace_root() -> Path:
    return resolve_existing_aliases(
        [
            ("kinver",),
            ("KinVer-master", "KinVer-master"),
        ]
    )


def family_project_root() -> Path:
    return resolve_existing_aliases(
        [
            ("family",),
            ("Family-Kinship-Recognition--main", "Family-Kinship-Recognition--main"),
        ]
    )


def gae_project_root() -> Path:
    return kinface_workspace_root() / "Source Code Matlab" / "common" / "methods" / "gae"
