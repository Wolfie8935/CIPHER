"""
Dead drop system for CIPHER — the MEMENTO layer.

Implements the dead drop schema: write, read, verify integrity, and manage
the filesystem-based vault. Dead drops are RED's only memory bridge across
context resets.

Owns: dead drop data model, vault I/O, integrity hashing, obfuscated filenames.
Does NOT own: agent decision-making about WHEN to write drops, observation
generation, or BLUE's tampering mechanic (Phase 3).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeadDrop:
    """
    A structured dead drop — the unit of externalized memory in CIPHER.

    Contains exactly enough information for an incoming RED agent to continue
    a mission after a context reset. Over-writing is penalized via
    memory_efficiency_score. Under-writing causes mission failure.
    """

    dead_drop_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    written_by: str = ""
    written_at_step: int = 0
    schema_version: str = "1.0"

    mission_status: dict[str, Any] = field(default_factory=dict)
    environment_map: dict[str, Any] = field(default_factory=dict)
    suspicion_state: dict[str, Any] = field(default_factory=dict)
    traps_placed: list[dict[str, Any]] = field(default_factory=list)
    continuation_directive: str = ""

    integrity_hash: str = ""
    token_count: int = 0

    def compute_hash(self) -> str:
        """
        Compute SHA-256 of the drop contents, excluding the integrity_hash field.

        Returns:
            The hex digest of the SHA-256 hash.
        """
        data = self._hashable_dict()
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """
        Verify that integrity_hash matches the computed hash.

        Returns:
            True if the hash matches (drop is untampered). False = tampered.
        """
        if not self.integrity_hash:
            return False
        return self.integrity_hash == self.compute_hash()

    def to_json(self) -> str:
        """Serialize the dead drop to a JSON string."""
        return json.dumps(self._full_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, json_str: str) -> DeadDrop:
        """
        Deserialize a DeadDrop from a JSON string.

        Args:
            json_str: A JSON string previously produced by to_json().

        Returns:
            A reconstructed DeadDrop instance.
        """
        data = json.loads(json_str)
        return cls(
            dead_drop_id=data.get("dead_drop_id", str(uuid.uuid4())),
            written_by=data.get("written_by", ""),
            written_at_step=data.get("written_at_step", 0),
            schema_version=data.get("schema_version", "1.0"),
            mission_status=data.get("mission_status", {}),
            environment_map=data.get("environment_map", {}),
            suspicion_state=data.get("suspicion_state", {}),
            traps_placed=data.get("traps_placed", []),
            continuation_directive=data.get("continuation_directive", ""),
            integrity_hash=data.get("integrity_hash", ""),
            token_count=data.get("token_count", 0),
        )

    def estimate_token_count(self) -> int:
        """
        Rough token estimate: len(json_str) / 4.

        Returns:
            Estimated number of tokens in this drop.
        """
        return max(1, len(self.to_json()) // 4)

    def _hashable_dict(self) -> dict[str, Any]:
        """Return dict of all fields EXCEPT integrity_hash, for hashing."""
        d = self._full_dict()
        d.pop("integrity_hash", None)
        return d

    def _full_dict(self) -> dict[str, Any]:
        """Return the complete dict representation of this drop."""
        return {
            "dead_drop_id": self.dead_drop_id,
            "written_by": self.written_by,
            "written_at_step": self.written_at_step,
            "schema_version": self.schema_version,
            "mission_status": self.mission_status,
            "environment_map": self.environment_map,
            "suspicion_state": self.suspicion_state,
            "traps_placed": self.traps_placed,
            "continuation_directive": self.continuation_directive,
            "integrity_hash": self.integrity_hash,
            "token_count": self.token_count,
        }


class DeadDropVault:
    """
    Manages the filesystem-level dead drop store.

    Drops are written to drop_vault/ with obfuscated filenames.
    BLUE can discover drops by searching (action budget cost).
    RED can read drops at their current node.

    Does NOT handle the BLUE tampering mechanic (Phase 3).
    Does NOT handle encryption (obfuscation via filename is sufficient for Phase 1).
    """

    def __init__(self, vault_dir: Path, max_tokens_per_drop: int) -> None:
        self.vault_dir = vault_dir
        self.max_tokens_per_drop = max_tokens_per_drop
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.vault_dir / "vault_index.json"
        self._index: dict[str, int] = self._load_index()

    def write(self, drop: DeadDrop, node_id: int) -> tuple[str, float]:
        """
        Write a dead drop to disk associated with a node.

        Computes and sets the integrity hash before writing.
        Truncates and logs a warning if token count exceeds max_tokens_per_drop.

        Args:
            drop: The DeadDrop to write.
            node_id: The node ID this drop is associated with.

        Returns:
            A tuple of (file_path, memory_efficiency_score).
            memory_efficiency_score = min(1.0, max_tokens / actual_tokens).
        """
        # Estimate tokens
        drop.token_count = drop.estimate_token_count()

        # Truncate continuation directive if too long
        if drop.token_count > self.max_tokens_per_drop:
            logger.warning(
                f"Dead drop {drop.dead_drop_id} exceeds max tokens "
                f"({drop.token_count} > {self.max_tokens_per_drop}). "
                f"Truncating continuation_directive."
            )
            # Crude truncation — cut the directive
            max_directive_chars = max(
                10,
                len(drop.continuation_directive)
                - (drop.token_count - self.max_tokens_per_drop) * 4,
            )
            drop.continuation_directive = drop.continuation_directive[
                :max_directive_chars
            ]
            drop.token_count = drop.estimate_token_count()

        # Compute integrity hash
        drop.integrity_hash = drop.compute_hash()

        # Generate obfuscated filename
        filename = self._obfuscate_filename(node_id, drop.dead_drop_id)
        filepath = self.vault_dir / filename

        # Write to disk
        filepath.write_text(drop.to_json(), encoding="utf-8")

        # Update index
        self._index[filename] = node_id
        self._save_index()

        # Compute memory efficiency
        actual_tokens = drop.token_count
        memory_efficiency = min(1.0, self.max_tokens_per_drop / max(1, actual_tokens))

        logger.debug(
            f"Dead drop written: {filename} at node {node_id} "
            f"(tokens={actual_tokens}, efficiency={memory_efficiency:.2f})"
        )

        return str(filepath), memory_efficiency

    def read(self, node_id: int) -> list[DeadDrop]:
        """
        Read all dead drops available at a given node.

        Verifies integrity hash on each drop. Tampered drops are included
        but flagged — call drop.verify() to check.

        Args:
            node_id: The node to read drops from.

        Returns:
            List of DeadDrop objects at this node.
        """
        drops: list[DeadDrop] = []

        for filename, mapped_node in self._index.items():
            if mapped_node != node_id:
                continue

            filepath = self.vault_dir / filename
            if not filepath.exists():
                logger.warning(f"Indexed drop file missing: {filepath}")
                continue

            try:
                json_str = filepath.read_text(encoding="utf-8")
                drop = DeadDrop.from_json(json_str)

                if not drop.verify():
                    logger.warning(
                        f"Dead drop {drop.dead_drop_id} at node {node_id} "
                        f"FAILED integrity check — possible tampering"
                    )

                drops.append(drop)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error(f"Failed to read drop {filepath}: {exc}")

        return drops

    def list_all_drop_paths(self) -> list[str]:
        """
        Return all drop file paths in the vault.

        Used by BLUE's Threat Hunter to enumerate discoverable drops.

        Returns:
            List of absolute file path strings.
        """
        return [
            str(self.vault_dir / filename)
            for filename in self._index
            if (self.vault_dir / filename).exists()
        ]

    def get_drops_at_node(self, node_id: int) -> list[str]:
        """Returns file paths of all drops at a given node. Uses vault_index."""
        return [
            str(self.vault_dir / filename)
            for filename, mapped_node in self._index.items()
            if mapped_node == node_id and (self.vault_dir / filename).exists()
        ]

    def clear(self) -> None:
        """
        Clear all drops from the vault. Called at episode start.

        Removes all .drop files and resets the index.
        On Windows-mounted or read-only filesystems, deletion may be
        silently skipped (the index is still reset so the drops are
        logically invisible to the new episode).
        """
        for filepath in self.vault_dir.glob("*.drop"):
            try:
                filepath.unlink()
            except (PermissionError, OSError):
                # On NTFS-mounted paths the file may not be unlinkable;
                # reset the index anyway so old drops are invisible.
                pass

        self._index = {}
        self._save_index()
        logger.debug("Dead drop vault cleared")

    def _obfuscate_filename(self, node_id: int, drop_id: str) -> str:
        """
        Generate an obfuscated filename encoding the node_id.

        Uses SHA-256 of "{node_id}:{drop_id}" truncated to 16 hex chars.
        BLUE cannot trivially enumerate drops by node — they must discover paths.

        Args:
            node_id: The node the drop is placed at.
            drop_id: The drop's unique ID.

        Returns:
            An obfuscated filename ending in .drop.
        """
        raw = f"{node_id}:{drop_id}"
        hashed = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{hashed}.drop"

    def _node_id_from_path(self, path: Path) -> int | None:
        """
        Reverse-lookup: given a drop file path, return the node_id.

        Uses the vault index to map filenames to node IDs.

        Args:
            path: Path to a .drop file.

        Returns:
            The node_id, or None if not found in the index.
        """
        filename = path.name
        return self._index.get(filename)

    def _load_index(self) -> dict[str, int]:
        """Load the vault index from disk, or return empty dict."""
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                return {str(k): int(v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(f"Failed to load vault index: {exc}")
        return {}

    def _save_index(self) -> None:
        """Save the vault index to disk."""
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )


def build_dead_drop_from_state(
    agent_id: str,
    state: "EpisodeState",
    red_obs: "RedObservation",
    directive: str,
) -> DeadDrop:
    """
    Construct a DeadDrop from the current episode state and RED observation.

    This is the helper that RED agents call when they decide to WRITE_DEAD_DROP.
    They don't fill the schema manually — they call this function.

    Args:
        agent_id: The writing agent's ID.
        state: The current ground-truth episode state.
        red_obs: The RED agent's current observation.
        directive: The continuation directive text.

    Returns:
        A fully populated DeadDrop ready to be written to the vault.
    """
    # Avoid circular imports
    from cipher.environment.state import EpisodeState
    from cipher.environment.observation import RedObservation

    drop = DeadDrop(
        written_by=agent_id,
        written_at_step=state.step,
        mission_status={
            "phase": "lateral_movement",
            "current_node": state.red_current_node,
            "exfiltrated_files": list(state.red_exfiltrated_files),
            "steps_elapsed": state.step,
        },
        environment_map={
            "confirmed_visited": list(set(state.red_visited_nodes)),
            "current_adjacent": red_obs.adjacent_nodes,
            "files_at_current": red_obs.files_at_current_node,
        },
        suspicion_state={
            "current_estimated_score": red_obs.estimated_suspicion,
            "context_resets_survived": state.red_context_resets,
        },
        traps_placed=list(state.red_traps_placed),
        continuation_directive=directive,
    )

    return drop
