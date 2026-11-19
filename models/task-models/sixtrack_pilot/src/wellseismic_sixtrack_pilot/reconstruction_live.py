"""Internal cancellation-aware live preparation; no public plugin entry point."""
from pathlib import Path

from .manifest import sha256_file
from .runner import _invoke_worker


def run_live_reconstruction(*, volume_path: Path, volume_sha256: str,
                            support_path: Path, support_sha256: str,
                            source_root: Path, snapshot: Path, output_directory: Path) -> dict:
    """Trusted callers authorize all sources first; worker verifies content identities."""
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        raise ValueError("Use a new output directory")
    for path, checksum in ((volume_path, volume_sha256), (support_path, support_sha256)):
        if sha256_file(Path(path)) != checksum:
            raise ValueError("Live P21 input hash mismatch")
    result = _invoke_worker("wellseismic_sixtrack_pilot.worker.reconstruction_live", {
        "volume_path": str(Path(volume_path).resolve()), "volume_sha256": volume_sha256,
        "support_path": str(Path(support_path).resolve()), "support_sha256": support_sha256,
        "source_root": str(Path(source_root).resolve()), "snapshot": str(Path(snapshot).resolve()),
        "output_directory": str(output_directory / "live"),
    }, output_directory)
    if sha256_file(Path(result["outputs"]["predictions_npz"])) != result["output_sha256"]:
        raise RuntimeError("Live P21 output hash mismatch")
    return result
