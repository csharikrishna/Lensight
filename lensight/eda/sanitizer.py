"""Dataset Sanitizer and Remediation Utilities.

Provides non-destructive dataset cleaning (e.g. creating in-memory PyTorch Subsets),
safe directory exporting, and controlled in-place cleaning.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, List, Optional, Sequence, Set

from .auditor import AuditReport, LeakageReport


class DatasetSanitizer:
    """Utilities for cleaning datasets based on audit and leakage findings."""

    @staticmethod
    def clean_subset(
        dataset: Any,
        audit_report: AuditReport,
        remove_duplicates: bool = True,
        remove_cross_label: bool = True,
    ) -> Any:
        """Create an in-memory PyTorch Subset containing only clean, non-redundant samples.

        Zero disk modification. Preserves the original dataset completely untouched.

        Args:
            dataset: The PyTorch Dataset instance.
            audit_report: AuditReport obtained from DatasetAuditor or audit_dataset.
            remove_duplicates: Whether to drop redundant duplicate copies (keeps best survivor).
            remove_cross_label: Whether to drop ambiguous cross-label conflicting samples.

        Returns:
            torch.utils.data.Subset with clean indices.
        """
        try:
            from torch.utils.data import Subset
        except ImportError:
            raise ImportError("PyTorch is required to create a torch.utils.data.Subset.")

        clean_indices = audit_report.get_clean_indices(
            remove_duplicates=remove_duplicates,
            remove_cross_label=remove_cross_label,
        )
        return Subset(dataset, clean_indices)

    @staticmethod
    def clean_test_subset(test_dataset: Any, leakage_report: LeakageReport) -> Any:
        """Create a leakage-free PyTorch Subset of test_dataset."""
        try:
            from torch.utils.data import Subset
        except ImportError:
            raise ImportError("PyTorch is required to create a torch.utils.data.Subset.")

        clean_indices = leakage_report.get_clean_test_indices()
        return Subset(test_dataset, clean_indices)

    @staticmethod
    def export_clean_directory(
        source_dir: str,
        output_dir: str,
        audit_report: AuditReport,
        mode: str = "copy",
        remove_duplicates: bool = True,
        remove_cross_label: bool = True,
    ) -> int:
        """Export clean files from source_dir to output_dir mirroring class subfolder structure.

        Args:
            source_dir: Root path of source dataset with class subfolders.
            output_dir: Target destination path for clean dataset.
            audit_report: AuditReport from audit_dataset(source_dir).
            mode: 'copy' (safe copy) or 'move' (relocate files).
            remove_duplicates: Exclude redundant duplicates (keeping best survivor).
            remove_cross_label: Exclude conflicting cross-label samples.

        Returns:
            Number of clean files written to output_dir.
        """
        if mode not in ("copy", "move"):
            raise ValueError("mode must be 'copy' or 'move'")
        return audit_report.export_clean_dataset(
            output_dir=output_dir,
            mode=mode,
            remove_duplicates=remove_duplicates,
            remove_cross_label=remove_cross_label,
        )

    @staticmethod
    def clean_inplace(
        source_dir: str,
        audit_report: AuditReport,
        confirm: bool = False,
        backup_manifest_path: Optional[str] = None,
        remove_duplicates: bool = True,
        remove_cross_label: bool = True,
    ) -> List[str]:
        """In-place removal of redundant duplicates and cross-label conflicts from disk.

        Requires explicit confirm=True for safety.

        Args:
            source_dir: Root dataset folder on disk.
            audit_report: AuditReport generated on source_dir.
            confirm: Must be True to execute destructive deletion.
            backup_manifest_path: Optional path to save a JSON log of deleted files.
            remove_duplicates: Whether to remove duplicate copies.
            remove_cross_label: Whether to remove conflicting cross-label copies.

        Returns:
            List of deleted file paths.
        """
        if not confirm:
            raise RuntimeError(
                "Destructive in-place cleaning was aborted. Set confirm=True to proceed with deletion."
            )

        files_to_delete: Set[str] = set()
        if remove_duplicates:
            for cluster in audit_report.duplicate_clusters:
                for dup_id in cluster.duplicate_ids:
                    if isinstance(dup_id, str) and os.path.isfile(dup_id):
                        files_to_delete.add(dup_id)

        if remove_cross_label:
            for conf in audit_report.cross_label_conflicts:
                if isinstance(conf.sample_b_id, str) and os.path.isfile(conf.sample_b_id):
                    files_to_delete.add(conf.sample_b_id)

        deleted: List[str] = []
        for p in files_to_delete:
            try:
                os.remove(p)
                deleted.append(p)
            except Exception as e:
                print(f"Warning: could not delete {p}: {e}")

        if backup_manifest_path:
            with open(backup_manifest_path, "w", encoding="utf-8") as f:
                json.dump({"deleted_files": deleted, "count": len(deleted)}, f, indent=2)

        return deleted
