import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dvc_data(paths: list[str | Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if not missing:
        logger.info("All %d DVC data files present", len(paths))
        return

    logger.info("Downloading %d missing DVC file(s): %s", len(missing), missing)
    result = subprocess.run(
        ["dvc", "pull"] + missing,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dvc pull failed (exit code {result.returncode}): {result.stderr}"
        )
    logger.info("DVC pull completed successfully")
