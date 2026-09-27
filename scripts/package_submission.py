"""
Final Submission Packaging Script
=================================
Assembles and validates the official competition package:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv      # final matches (scored on leaderboard)
│   └── candidate_pairs.tsv       # candidate set from blocking
├── code/
│   └── business_entity_resolution/
│       ├── src/                  # all source code
│       ├── README.md             # end-to-end execution guide
│       └── requirements.txt      # pinned dependencies
└── Documentation_template.md     # methodology write-up
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package_submission(
    matching_file: Path,
    candidate_file: Path,
    output_zip: Path,
    documentation_file: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    skip_validation: bool = False,
) -> None:
    # 1. Validation
    if not matching_file.exists():
        raise FileNotFoundError(f"matching_results.tsv not found at {matching_file}")
    if not candidate_file.exists():
        raise FileNotFoundError(f"candidate_pairs.tsv not found at {candidate_file}")

    if not skip_validation:
        print("=" * 70)
        print("[1/3] Running official submission validator...")
        print("=" * 70)
        val_script = PROJECT_ROOT / "utils" / "validate_submission.py"
        t_dir = test_dir or (PROJECT_ROOT / "dataset" / "test")
        val_cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(matching_file),
            "--candidate", str(candidate_file),
            "--test-dir", str(t_dir),
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"Validator failed with exit code {res.returncode}. Aborting packaging.")

    # 2. Package into zip
    print("=" * 70)
    print(f"[2/3] Assembling submission archive -> {output_zip}...")
    print("=" * 70)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    code_dir = PROJECT_ROOT / "code" / "business_entity_resolution"

    doc_path = documentation_file or (PROJECT_ROOT / "Documentation_template.md")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # A. output/ folder
        print(f"  + output/matching_results.tsv ({matching_file.stat().st_size:,} bytes)")
        zf.write(matching_file, arcname="output/matching_results.tsv")

        print(f"  + output/candidate_pairs.tsv ({candidate_file.stat().st_size:,} bytes)")
        zf.write(candidate_file, arcname="output/candidate_pairs.tsv")

        # B. code/business_entity_resolution/
        # Add src/
        src_dir = code_dir / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith((".py", ".json", ".yaml", ".yml", ".md", ".txt")) and not f.endswith((".pyc", ".pyo")):
                        fp = Path(root) / f
                        arcname = f"code/business_entity_resolution/src/{fp.relative_to(src_dir).as_posix()}"
                        print(f"  + {arcname}")
                        zf.write(fp, arcname=arcname)

        # Add README.md and requirements.txt
        for extra_f in ("README.md", "requirements.txt"):
            fp = code_dir / extra_f
            if fp.exists():
                arcname = f"code/business_entity_resolution/{extra_f}"
                print(f"  + {arcname}")
                zf.write(fp, arcname=arcname)
            else:
                # check project root fallback
                fp_root = PROJECT_ROOT / extra_f
                if fp_root.exists():
                    arcname = f"code/business_entity_resolution/{extra_f}"
                    print(f"  + {arcname} (from project root)")
                    zf.write(fp_root, arcname=arcname)

        # C. Documentation_template.md
        if doc_path.exists():
            print(f"  + {doc_path.name}")
            zf.write(doc_path, arcname=doc_path.name)
        else:
            print("  ! Documentation_template.md not found, creating placeholder")
            zf.writestr("Documentation_template.md", "# Business Entity Resolution Methodology\n")

    # 3. Verify Zip contents
    print("=" * 70)
    print(f"[3/3] Verifying package integrity: {output_zip} ({output_zip.stat().st_size:,} bytes)")
    print("=" * 70)
    with zipfile.ZipFile(output_zip, "r") as zf:
        infolist = zf.infolist()
        print(f"Total files in archive: {len(infolist)}")
        for info in infolist:
            print(f"  - {info.filename:<55} {info.file_size:>10,} bytes")

    print("\n[SUCCESS] Submission package successfully assembled and verified!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package official competition submission zip")
    parser.add_argument("--matching", type=Path, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--output-zip", type=Path, default=Path("submissions/submission.zip"), help="Output zip path")
    parser.add_argument("--documentation", type=Path, default=None, help="Path to Documentation_template.md")
    parser.add_argument("--test-dir", type=Path, default=None, help="Path to dataset/test directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running validator")
    args = parser.parse_args()

    package_submission(
        matching_file=args.matching,
        candidate_file=args.candidate,
        output_zip=args.output_zip,
        documentation_file=args.documentation,
        test_dir=args.test_dir,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
