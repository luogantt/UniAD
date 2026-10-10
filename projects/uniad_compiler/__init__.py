"""Two-environment compiler data boundary for UniAD 2.0.

The package itself requires only NumPy.  Checkpoint extraction runs in the
official UniAD environment, while :mod:`artifact` emits a verified data-only
bundle that a separate GAC environment can consume.
"""

from .artifact import (
    ArtifactError,
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    UniADFFNBundle,
    load_bundle,
    write_bundle,
)
from .checkpoint import (
    CANONICAL_WEIGHT_SHAPES,
    CheckpointContractError,
    ENCODER_LAYER_COUNT,
    encoder_ffn_checkpoint_keys,
    extract_encoder_ffn_weights,
)
from .contract import (
    CANONICAL_WEIGHT_NAMES,
    CONTRACT_KIND,
    CONTRACT_SCHEMA_VERSION,
    ContractError,
    SAMPLE_NAMES,
    UniADFFNContract,
)


__all__ = [
    "ArtifactError",
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "CANONICAL_WEIGHT_NAMES",
    "CANONICAL_WEIGHT_SHAPES",
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "CheckpointContractError",
    "ContractError",
    "ENCODER_LAYER_COUNT",
    "SAMPLE_NAMES",
    "UniADFFNBundle",
    "UniADFFNContract",
    "encoder_ffn_checkpoint_keys",
    "extract_encoder_ffn_weights",
    "load_bundle",
    "write_bundle",
]
