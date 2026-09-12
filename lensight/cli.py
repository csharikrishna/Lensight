"""Command Line Interface for Lensight Dataset Health, Leakage & Sanitization."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .eda.auditor import DatasetAuditor, check_leakage
from .eda.sanitizer import DatasetSanitizer


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="lensight",
        description="Lensight: Computer Vision Model Diagnostics & Dataset Auditing Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: audit
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit a dataset for duplicates, class imbalance, corrupt files, and cross-label errors",
    )
    audit_parser.add_argument("--dataset", required=True, help="Path to dataset root folder with class subdirectories")
    audit_parser.add_argument("--report", default="audit_report.html", help="Path to export interactive HTML dashboard")
    audit_parser.add_argument("--json", default=None, help="Optional path to export JSON metrics")
    audit_parser.add_argument("--threshold", type=int, default=3, help="Hamming distance threshold for near-duplicates (default: 3)")
    audit_parser.add_argument("--algo", choices=["dhash", "ahash", "phash"], default="dhash", help="Perceptual hash algorithm (default: dhash)")
    audit_parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples to scan")

    # Command: leakage
    leakage_parser = subparsers.add_parser(
        "leakage",
        help="Detect train/test data leakage across splits",
    )
    leakage_parser.add_argument("--train", required=True, help="Path to train split folder with class subdirectories")
    leakage_parser.add_argument("--test", required=True, help="Path to test split folder with class subdirectories")
    leakage_parser.add_argument("--threshold", type=int, default=3, help="Hamming distance threshold for leakage (default: 3)")
    leakage_parser.add_argument("--algo", choices=["dhash", "ahash", "phash"], default="dhash", help="Perceptual hash algorithm (default: dhash)")
    leakage_parser.add_argument("--json", default=None, help="Optional path to export JSON leakage metrics")

    # Command: clean
    clean_parser = subparsers.add_parser(
        "clean",
        help="Sanitize dataset by removing redundant duplicate images and contradictory cross-label pairs",
    )
    clean_parser.add_argument("--dataset", required=True, help="Path to source dataset folder")
    clean_parser.add_argument("--output", default=None, help="Path to clean destination folder (required for mode='copy')")
    clean_parser.add_argument("--mode", choices=["copy", "inplace"], default="copy", help="Cleaning mode: 'copy' to new folder or destructive 'inplace'")
    clean_parser.add_argument("--confirm", action="store_true", help="Explicit confirmation required for mode='inplace'")
    clean_parser.add_argument("--threshold", type=int, default=3, help="Hamming distance threshold for duplicates (default: 3)")
    clean_parser.add_argument("--algo", choices=["dhash", "ahash", "phash"], default="dhash", help="Perceptual hash algorithm (default: dhash)")
    clean_parser.add_argument("--manifest", default=None, help="Optional path to write JSON file manifest")

    # Command: help
    subparsers.add_parser(
        "help",
        help="Print an interactive quick-reference cheat sheet for all workflows",
    )

    # Command: info
    subparsers.add_parser(
        "info",
        help="Display environment, PyTorch, CUDA, and package status",
    )

    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        sys.exit(1)

    if parsed.command == "help":
        from . import help as print_help
        print_help()
        return

    elif parsed.command == "info":
        import torch
        print("Lensight Environment Status:")
        print(f"  * Python:       {sys.version.split()[0]}")
        print(f"  * Lensight:     0.2.0")
        print(f"  * PyTorch:      {torch.__version__}")
        print(f"  * CUDA device:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (No CUDA detected)'}")
        return

    if parsed.command == "audit":
        print(f"[*] Auditing dataset: {parsed.dataset}")
        auditor = DatasetAuditor(max_samples=parsed.max_samples)
        report = auditor.audit(parsed.dataset, threshold=parsed.threshold, algo=parsed.algo)
        print("\n" + report.summary_text())

        if parsed.report:
            saved_path = report.save_html(parsed.report)
            print(f"\n[OK] Interactive HTML audit dashboard written to: {saved_path}")

        if parsed.json:
            with open(parsed.json, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"[OK] JSON metrics exported to: {parsed.json}")

    elif parsed.command == "leakage":
        print(f"[*] Checking leakage between:")
        print(f"    Train: {parsed.train}")
        print(f"    Test:  {parsed.test}")
        report = check_leakage(parsed.train, parsed.test, threshold=parsed.threshold, algo=parsed.algo)
        print("\n" + report.summary_text())

        if parsed.json:
            with open(parsed.json, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"[OK] JSON metrics exported to: {parsed.json}")

    elif parsed.command == "clean":
        print(f"[*] Auditing dataset before cleaning: {parsed.dataset}")
        auditor = DatasetAuditor()
        report = auditor.audit(parsed.dataset, threshold=parsed.threshold, algo=parsed.algo)
        print(f"Found {len(report.duplicate_clusters)} duplicate clusters ({report.total_duplicates_removed_count} redundant images).")

        if parsed.mode == "copy":
            if not parsed.output:
                print("[Error] --output is required when --mode is 'copy'")
                sys.exit(1)
            print(f"[*] Copying clean survivors to: {parsed.output} ...")
            copied = DatasetSanitizer.export_clean_directory(parsed.dataset, parsed.output, report, mode="copy")
            print(f"[OK] Successfully copied {copied} clean images into: {parsed.output}")
        elif parsed.mode == "inplace":
            if not parsed.confirm:
                print("[Error] In-place deletion is destructive. Re-run with --confirm to proceed.")
                sys.exit(1)
            deleted = DatasetSanitizer.clean_inplace(parsed.dataset, report, confirm=True)
            print(f"[OK] Deleted {len(deleted)} redundant duplicate files from disk.")

        if parsed.manifest:
            report.export_manifest(parsed.manifest)
            print(f"[OK] Audit manifest written to: {parsed.manifest}")


if __name__ == "__main__":
    main()
