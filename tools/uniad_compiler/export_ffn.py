#!/usr/bin/env python3
"""Export one UniAD 2.0 BEV encoder FFN into the compiler interchange bundle.

Run this command in the official UniAD Python 3.9 / Torch 2.0 environment.
The output contains canonical NumPy weights and a checksummed JSON contract;
it deliberately does not import the Python 3.11 General AI Compiler runtime.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.uniad_compiler import (  # noqa: E402
    UniADFFNContract,
    encoder_ffn_checkpoint_keys,
    extract_encoder_ffn_weights,
    write_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    for name in ("state_dict", "model"):
        candidate = checkpoint.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    return checkpoint


def _selected_numpy_state_dict(state_dict: Mapping[str, Any], layer: int) -> dict[str, np.ndarray]:
    required = set(encoder_ffn_checkpoint_keys(layer).values())
    selected: dict[str, np.ndarray] = {}
    for raw_name, value in state_dict.items():
        if not isinstance(raw_name, str):
            continue
        name = raw_name[7:] if raw_name.startswith("module.") else raw_name
        if name not in required:
            continue
        if not hasattr(value, "detach"):
            raise TypeError(f"checkpoint value {raw_name!r} is not a Torch tensor")
        selected[name] = value.detach().cpu().numpy()
    return selected


def _torch_golden(
    query: np.ndarray,
    weights: Mapping[str, np.ndarray],
    epsilon: float,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        value = torch.from_numpy(query)
        residual = value
        value = functional.linear(
            value,
            torch.from_numpy(weights["fc1_weight"].T.copy()),
            torch.from_numpy(weights["fc1_bias"]),
        )
        value = functional.relu(value)
        value = functional.linear(
            value,
            torch.from_numpy(weights["fc2_weight"].T.copy()),
            torch.from_numpy(weights["fc2_bias"]),
        )
        value = functional.layer_norm(
            residual + value,
            (query.shape[-1],),
            torch.from_numpy(weights["norm_weight"]),
            torch.from_numpy(weights["norm_bias"]),
            epsilon,
        )
    return np.ascontiguousarray(value.numpy(), dtype=np.float32)


def _load_checkpoint(path: Path) -> Any:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # Torch 2.0 in the documented UniAD environment predates the stable
        # weights_only loader.  Checkpoints must therefore come from a trusted
        # source when that environment is used.
        return torch.load(str(path), map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=0, choices=range(6))
    parser.add_argument(
        "--with-synthetic-golden",
        action="store_true",
        help=(
            "also export a deterministic [1,40000,256] query and Torch eager "
            "golden output (about 82 MiB before NPZ overhead)"
        ),
    )
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")

    contract = UniADFFNContract(layer=args.layer)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _selected_numpy_state_dict(
        _unwrap_state_dict(checkpoint), args.layer
    )
    weights = extract_encoder_ffn_weights(state_dict, args.layer)

    query = None
    golden = None
    if args.with_synthetic_golden:
        generator = np.random.RandomState(args.seed)
        query = np.ascontiguousarray(
            generator.normal(0.0, 0.1, contract.input_shape).astype(np.float32)
        )
        golden = _torch_golden(query, weights, contract.layer_norm_epsilon)

    write_bundle(
        args.output,
        contract,
        weights,
        sample=(
            {"query": query, "output": golden}
            if query is not None and golden is not None
            else None
        ),
    )
    bundle_path = args.output.resolve()
    provenance = {
        "schema_version": 1,
        "source": "UniAD 2.0 PyTorch checkpoint",
        "bundle_file": bundle_path.name,
        "bundle_sha256": _sha256(bundle_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_key_prefix": (
            "pts_bbox_head.transformer.encoder.layers.%d" % args.layer
        ),
        "golden_runtime": (
            "torch-eager-cpu" if args.with_synthetic_golden else None
        ),
        "synthetic_seed": args.seed if args.with_synthetic_golden else None,
    }
    provenance_path = Path(str(bundle_path) + ".provenance.json")
    import json

    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    print("bundle_sha256=" + provenance["bundle_sha256"])
    print("provenance=" + str(provenance_path))


if __name__ == "__main__":
    main()
