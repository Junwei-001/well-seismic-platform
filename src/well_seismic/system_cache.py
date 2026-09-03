"""Fail-closed cleanup for regenerable platform caches.

The cache API must never accept an arbitrary path from an HTTP caller.  This
module therefore works from a fixed, validated allow-list and deliberately
excludes source data, control-plane state, prediction products, reports and
model/runtime directories.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


CACHE_CONTRACT_VERSION = "well-seismic.system-cache.v1"

# A configured cache root may only be a separate top-level cache directory.
# These platform trees contain durable or source-controlled material and are
# never valid cache roots, even when a nested directory happens to contain the
# word "cache".
_PROTECTED_TOP_LEVEL_NAMES = frozenset(
    {
        ".git",
        ".github",
        "artifacts",
        "configs",
        "data",
        "docs",
        "frontend",
        "model_outputs",
        "models",
        "requirements",
        "runtime",
        "scripts",
        "src",
        "tests",
        "tools",
        "比赛输入示例_自动识别",
        "接口模型",
        "可视化界面",
        "离线环境",
        "设计审核",
        "输出结果",
    }
)


class CacheSafetyError(ValueError):
    """Raised when a cache root does not satisfy the deletion boundary."""


class ClearableMemoryCache(Protocol):
    """Small protocol shared by bounded in-process caches."""

    @property
    def stats(self) -> Mapping[str, int]: ...

    def clear(self) -> None: ...


def _within(path: Path, parent: Path) -> bool:
    try:
        common = os.path.commonpath(
            [os.path.normcase(str(path)), os.path.normcase(str(parent))]
        )
    except ValueError:
        return False
    return common == os.path.normcase(str(parent))


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _entry_is_link(entry: os.DirEntry[str], stat_result: os.stat_result) -> bool:
    return entry.is_symlink() or _is_reparse(stat_result)


class SystemCacheService:
    """Inspect and clear only explicitly registered, regenerable caches."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        filesystem_roots: Mapping[str, str | Path],
        memory_caches: Mapping[str, ClearableMemoryCache] | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self._filesystem_roots = {
            str(scope): self._validate_root_definition(path)
            for scope, path in filesystem_roots.items()
        }
        self._memory_caches = dict(memory_caches or {})
        if not self._filesystem_roots and not self._memory_caches:
            raise ValueError("at least one cache scope is required")
        self._lock = threading.RLock()

    @classmethod
    def for_project(
        cls,
        *,
        project_root: str | Path,
        memory_caches: Mapping[str, ClearableMemoryCache] | None = None,
    ) -> "SystemCacheService":
        project = Path(project_root).expanduser().resolve()
        return cls(
            project_root=project,
            # The runtime cache path is fixed in code.  A user-controlled path
            # override would turn a cache operation into an arbitrary recursive
            # delete primitive, even with heuristic name checks.
            filesystem_roots={"runtime_files": project / ".runtime-cache"},
            memory_caches=memory_caches,
        )

    def _validate_root_definition(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        # abspath normalises ``..`` without dereferencing a symlink.  Check it
        # before resolve() so a configured path cannot escape and re-enter.
        lexical = Path(os.path.abspath(str(candidate)))
        if lexical == self.project_root or not _within(lexical, self.project_root):
            raise CacheSafetyError("cache root must be a child of the platform root")
        relative = lexical.relative_to(self.project_root)
        if not relative.parts:
            raise CacheSafetyError("platform root cannot be used as a cache root")
        if relative.parts != (".runtime-cache",):
            raise CacheSafetyError(
                "filesystem cache root must be the fixed .runtime-cache directory"
            )
        if relative.parts[0].casefold() in _PROTECTED_TOP_LEVEL_NAMES:
            raise CacheSafetyError(
                f"protected platform tree cannot be used as cache: {relative.parts[0]}"
            )
        if "cache" not in lexical.name.casefold():
            raise CacheSafetyError("cache root name must contain 'cache'")
        if lexical.exists():
            root_stat = os.lstat(lexical)
            if lexical.is_symlink() or _is_reparse(root_stat):
                raise CacheSafetyError("cache root cannot be a link or reparse point")
        resolved = lexical.resolve()
        if not _within(resolved, self.project_root):
            raise CacheSafetyError("cache root resolves outside the platform root")
        return lexical

    def _validate_root_at_use(self, root: Path) -> None:
        if not root.exists():
            return
        root_stat = os.lstat(root)
        if root.is_symlink() or _is_reparse(root_stat):
            raise CacheSafetyError("cache root became a link or reparse point")
        if not root.is_dir():
            raise CacheSafetyError("cache root is not a directory")
        if not _within(root.resolve(), self.project_root):
            raise CacheSafetyError("cache root resolves outside the platform root")

    def _scan_tree(self, root: Path) -> dict[str, int]:
        counts = {"files": 0, "directories": 0, "bytes": 0}
        if not root.exists():
            return counts

        def visit(directory: Path) -> None:
            try:
                entries = list(os.scandir(directory))
            except FileNotFoundError:
                return
            for entry in entries:
                try:
                    item_stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if _entry_is_link(entry, item_stat):
                    counts["files"] += 1
                    counts["bytes"] += int(item_stat.st_size)
                elif entry.is_dir(follow_symlinks=False):
                    counts["directories"] += 1
                    visit(Path(entry.path))
                else:
                    counts["files"] += 1
                    counts["bytes"] += int(item_stat.st_size)

        visit(root)
        return counts

    def inspect(self) -> dict[str, Any]:
        """Return current cache usage without creating or changing anything."""

        with self._lock:
            scopes: list[dict[str, Any]] = []
            errors: list[dict[str, str]] = []
            totals = {"files": 0, "directories": 0, "bytes": 0, "memory_entries": 0}
            for scope, root in self._filesystem_roots.items():
                try:
                    self._validate_root_at_use(root)
                    counts = self._scan_tree(root)
                    scopes.append(
                        {
                            "scope": scope,
                            "storage": "filesystem",
                            "exists": root.is_dir(),
                            **counts,
                        }
                    )
                    totals["files"] += counts["files"]
                    totals["directories"] += counts["directories"]
                    totals["bytes"] += counts["bytes"]
                except (OSError, CacheSafetyError) as exc:
                    errors.append(
                        {"scope": scope, "error": f"{type(exc).__name__}: {exc}"}
                    )
            for scope, cache in self._memory_caches.items():
                stats = dict(cache.stats)
                entries = int(stats.get("entries", 0))
                size = int(stats.get("bytes", 0))
                scopes.append(
                    {
                        "scope": scope,
                        "storage": "memory",
                        "entries": entries,
                        "bytes": size,
                    }
                )
                totals["memory_entries"] += entries
                totals["bytes"] += size
            return {
                "contract_version": CACHE_CONTRACT_VERSION,
                "clear_endpoint": "/api/v1/system/cache/clear",
                "scopes": scopes,
                "totals": totals,
                "errors": errors,
                "protected": [
                    "source_data",
                    "task_state",
                    "predictions",
                    "reports",
                    "model_weights",
                ],
            }

    def _clear_tree(
        self, root: Path, scope: str
    ) -> tuple[dict[str, int], list[dict[str, str]]]:
        counts = {
            "files_removed": 0,
            "directories_removed": 0,
            "bytes_reclaimed": 0,
        }
        errors: list[dict[str, str]] = []
        if not root.exists():
            return counts, errors

        def remove_contents(directory: Path) -> None:
            try:
                entries = list(os.scandir(directory))
            except FileNotFoundError:
                return
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                try:
                    item_stat = entry.stat(follow_symlinks=False)
                    is_link = _entry_is_link(entry, item_stat)
                    is_directory = entry.is_dir(follow_symlinks=False)
                    if is_link:
                        # Never traverse symlinks/junctions.  Remove only the
                        # in-cache link itself, leaving its target untouched.
                        if is_directory:
                            os.rmdir(path)
                        else:
                            os.unlink(path)
                        counts["files_removed"] += 1
                        counts["bytes_reclaimed"] += int(item_stat.st_size)
                    elif is_directory:
                        remove_contents(path)
                        try:
                            os.rmdir(path)
                        except FileNotFoundError:
                            continue
                        counts["directories_removed"] += 1
                    else:
                        os.unlink(path)
                        counts["files_removed"] += 1
                        counts["bytes_reclaimed"] += int(item_stat.st_size)
                except FileNotFoundError:
                    # A concurrent deletion is already the desired state.
                    continue
                except OSError as exc:
                    errors.append(
                        {
                            "scope": scope,
                            "entry": relative,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        remove_contents(root)
        return counts, errors

    def clear(self) -> dict[str, Any]:
        """Clear every allow-listed cache scope and report the exact effect."""

        with self._lock:
            scopes: list[dict[str, Any]] = []
            errors: list[dict[str, str]] = []
            totals = {
                "files_removed": 0,
                "directories_removed": 0,
                "bytes_reclaimed": 0,
                "memory_entries_removed": 0,
            }
            for scope, root in self._filesystem_roots.items():
                try:
                    self._validate_root_at_use(root)
                    counts, scope_errors = self._clear_tree(root, scope)
                    errors.extend(scope_errors)
                    scopes.append({"scope": scope, "storage": "filesystem", **counts})
                    for key in (
                        "files_removed",
                        "directories_removed",
                        "bytes_reclaimed",
                    ):
                        totals[key] += counts[key]
                except (OSError, CacheSafetyError) as exc:
                    errors.append(
                        {"scope": scope, "error": f"{type(exc).__name__}: {exc}"}
                    )
            for scope, cache in self._memory_caches.items():
                try:
                    before = dict(cache.stats)
                    entries = int(before.get("entries", 0))
                    size = int(before.get("bytes", 0))
                    cache.clear()
                    scopes.append(
                        {
                            "scope": scope,
                            "storage": "memory",
                            "entries_removed": entries,
                            "bytes_reclaimed": size,
                        }
                    )
                    totals["memory_entries_removed"] += entries
                    totals["bytes_reclaimed"] += size
                except Exception as exc:  # noqa: BLE001 - report a partial clear safely
                    errors.append(
                        {"scope": scope, "error": f"{type(exc).__name__}: {exc}"}
                    )
            return {
                "contract_version": CACHE_CONTRACT_VERSION,
                "scopes": scopes,
                **totals,
                "errors": errors,
                "protected": [
                    "source_data",
                    "task_state",
                    "predictions",
                    "reports",
                    "model_weights",
                ],
            }


__all__ = [
    "CACHE_CONTRACT_VERSION",
    "CacheSafetyError",
    "SystemCacheService",
]
