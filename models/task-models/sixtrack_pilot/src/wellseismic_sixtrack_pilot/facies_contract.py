"""Stdlib-only contracts for the original P81 development task binding."""
import json
from pathlib import Path

from .manifest import sha256_file

SOURCE_LOCK_PATH = Path(__file__).with_name("source_locks") / "facies_p81_recovery_v1.json"
SOURCE_LOCK_SHA256 = "4d0e46d02193f206eb084c77be3de90a65caaaf0b65c1c9db277eca33f6aa575"
TASK_CLASSES = {"facies_f3": 10, "facies_penobscot": 8}
MODEL_IDS = {"facies_f3": "sixtrack_facies_f3_p81", "facies_penobscot": "sixtrack_facies_penobscot_p81_reference"}
OPTION_KEYS = {"task_id", "fold_id", "repeat_id", "input_sha256", "state_sha256", "recovery_receipt_sha256"}


def source_lock():
    if sha256_file(SOURCE_LOCK_PATH) != SOURCE_LOCK_SHA256:
        raise ValueError("P81 packaged source lock integrity mismatch")
    return json.loads(SOURCE_LOCK_PATH.read_text())


def validate_options(options):
    if set(options) != OPTION_KEYS:
        raise ValueError("P81 requires only task/fold/repeat and three artifact hashes")
    if not isinstance(options["task_id"], str) or options["task_id"] not in TASK_CLASSES:
        raise ValueError("P81 F3 and Penobscot are independent development tasks")
    for key, limit in (("fold_id", 5), ("repeat_id", 3)):
        if type(options[key]) is not int or options[key] not in range(limit):
            raise ValueError(f"P81 {key} must be an original development integer")
    for key in ("input_sha256", "state_sha256", "recovery_receipt_sha256"):
        value = options[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"P81 {key} must be 64 lowercase hex characters")


def validate_binding(source, state, receipt_path, options):
    validate_options(options)
    for path, key in ((source, "input_sha256"), (state, "state_sha256"), (receipt_path, "recovery_receipt_sha256")):
        if sha256_file(Path(path)) != options[key]:
            raise ValueError(f"P81 {key} mismatch")
    receipt = json.loads(Path(receipt_path).read_text())
    lock = source_lock()
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != "well-seismic.p81-trained-state-recovery.v1"
            or receipt.get("status") != "completed" or receipt.get("source_lock_sha256") != SOURCE_LOCK_SHA256
            or receipt.get("query_targets_exported") is not False or receipt.get("frozen_test_accessed") is not False):
        raise ValueError("P81 recovery provenance is not an authorized development binding")
    for key in ("task_id", "fold_id", "repeat_id"):
        if type(receipt.get(key)) is not type(options[key]) or receipt[key] != options[key]:
            raise ValueError(f"P81 recovery {key} identity mismatch")
    task = options["task_id"]
    if (receipt.get("seed") != lock["seeds"][options["repeat_id"]]
            or receipt.get("num_classes") != TASK_CLASSES[task] or receipt.get("query_rows") != 16
            or receipt.get("label_version") != lock["label_versions"][task]
            or receipt.get("arm") != lock["arm"]):
        raise ValueError("P81 recovery model or label space identity mismatch")
    payload = receipt.get("payload_sha256")
    if (not isinstance(payload, dict) or payload.get("query.npz") != options["input_sha256"]
            or payload.get("model_state.pt") != options["state_sha256"]):
        raise ValueError("P81 query/state do not belong to the requested recovery")
    return receipt
