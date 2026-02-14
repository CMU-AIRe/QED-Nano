import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

try:
    from huggingface_hub import create_branch, create_repo, upload_folder
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:  # pragma: no cover - dependency provided by transformers
    create_branch = None  # type: ignore
    create_repo = None  # type: ignore
    upload_folder = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore
    _import_error = exc
else:
    _import_error = None

try:  # pragma: no cover - optional dependency exposed via huggingface_hub
    import requests
except Exception:  # pragma: no cover
    requests = None

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 5.0
_DEFAULT_MAX_DELAY = 60.0
_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _get_config_value(cfg, key: str, default=None):
    """Safely read a key from DictConfig or mapping-like objects."""
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        return getattr(cfg, key, default)


def format_revision(prefix: str | None, step: int) -> str:
    base = (prefix or "checkpoint").strip()
    if not base:
        base = "checkpoint"
    return f"{base}-step-{step:06d}"


def push_checkpoint_to_hub(
    cfg,
    checkpoint_dir: Path,
    step: int,
    *,
    extra_ignore: Sequence[str] | None = None,
) -> Future | None:
    """Upload a checkpoint directory to the Hugging Face Hub.

    Returns a Future when uploads run asynchronously, otherwise None.
    """
    if not bool(_get_config_value(cfg, "push_to_hub", False)):
        return None

    if _import_error:
        log.error("huggingface_hub is required for push_to_hub but could not be imported: %s", _import_error)
        return None

    hub_model_id = _get_config_value(cfg, "hub_model_id")
    if not hub_model_id:
        log.warning("push_to_hub enabled but hub_model_id is not set; skipping upload")
        return None

    if not checkpoint_dir.exists():
        log.warning("Checkpoint directory %s does not exist; skipping upload", checkpoint_dir)
        return None

    private = bool(_get_config_value(cfg, "hub_private", True))
    branch_prefix = _get_config_value(cfg, "hub_model_revision") or "checkpoint"
    branch_parent = _get_config_value(cfg, "hub_base_revision", "main")

    ignore_patterns: list[str] = []
    if extra_ignore:
        ignore_patterns.extend(extra_ignore)
    config_ignores: Iterable[str] | None = _get_config_value(cfg, "hub_ignore_patterns")
    if config_ignores:
        ignore_patterns.extend(config_ignores)

    max_retries = int(_get_config_value(cfg, "hub_max_retries", _DEFAULT_MAX_RETRIES))
    base_delay = float(_get_config_value(cfg, "hub_retry_base_seconds", _DEFAULT_BASE_DELAY))
    max_delay = float(_get_config_value(cfg, "hub_retry_max_seconds", _DEFAULT_MAX_DELAY))

    # Deduplicate while preserving order
    seen: set[str] = set()
    ignore_patterns = [pattern for pattern in ignore_patterns if not (pattern in seen or seen.add(pattern))]

    create_repo(
        repo_id=hub_model_id,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    revision = format_revision(branch_prefix, step)
    create_branch_kwargs = dict(
        repo_id=hub_model_id,
        branch=revision,
        repo_type="model",
        exist_ok=True,
    )
    if branch_parent:
        create_branch_kwargs["revision"] = branch_parent

    try:
        create_branch(**create_branch_kwargs)
    except HfHubHTTPError as err:
        # `exist_ok=True` still raises if revision/branch mismatch; surface once.
        if err.response is not None and err.response.status_code not in (409, 422):
            raise
        log.debug("Hub branch %s already exists on %s", revision, hub_model_id)

    commit_message = f"Add checkpoint {revision}"

    future = _UPLOAD_EXECUTOR.submit(
        _upload_with_retries,
        hub_model_id,
        str(checkpoint_dir),
        revision,
        commit_message,
        ignore_patterns or None,
        max_retries,
        base_delay,
        max_delay,
    )

    setattr(future, "_hf_revision", revision)
    log.info("Started Hub upload for %s@%s from %s", hub_model_id, revision, checkpoint_dir)
    return future


def _upload_with_retries(
    repo_id: str,
    folder_path: str,
    revision: str,
    commit_message: str,
    ignore_patterns: Sequence[str] | None,
    max_retries: int,
    base_delay: float,
    max_delay: float,
):
    """Run upload_folder with exponential backoff and jitter."""
    delay = max(base_delay, 0.0)
    attempts = max(1, max_retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                repo_type="model",
                revision=revision,
                commit_message=commit_message,
                ignore_patterns=ignore_patterns,
                run_as_future=False,
            )
            log.info("Completed Hub upload for %s@%s from %s", repo_id, revision, folder_path)
            return
        except Exception as err:  # noqa: BLE001
            last_error = err
            should_retry = attempt < attempts and _is_retryable_error(err)
            if not should_retry:
                log.error(
                    "Hub upload failed for %s@%s on attempt %d/%d: %s",
                    repo_id,
                    revision,
                    attempt,
                    attempts,
                    err,
                )
                raise

            sleep_for = min(delay if delay > 0 else base_delay, max_delay)
            # Apply jitter to avoid synchronized retries
            sleep_for *= random.uniform(0.8, 1.2)
            sleep_for = min(sleep_for, max_delay)
            log.warning(
                "Hub upload attempt %d/%d failed for %s@%s (%s). Retrying in %.1fs",
                attempt,
                attempts,
                repo_id,
                revision,
                err,
                sleep_for,
            )
            time.sleep(max(0.0, sleep_for))
            delay = min(delay * 2 if delay else base_delay * 2, max_delay)

    if last_error is not None:
        raise last_error


def _is_retryable_error(err: Exception) -> bool:
    if isinstance(err, HfHubHTTPError):
        response = getattr(err, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            return True
        return int(status) in _RETRYABLE_STATUS_CODES

    if requests is not None:
        if isinstance(err, (requests.ConnectionError, requests.Timeout)):  # type: ignore[has-type]
            return True
        if isinstance(err, requests.HTTPError):  # type: ignore[has-type]
            response = err.response
            status = getattr(response, "status_code", None)
            if status is None:
                return False
            return int(status) in _RETRYABLE_STATUS_CODES

    return False
