"""Packaging utility to bundle SmartTrack-AI into a clean, distributable ZIP archive."""

import os
import zipfile
from pathlib import Path


def package_project(output_zip: str = "SmartTrack-AI-Submission.zip") -> str:
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir.parent / output_zip

    # Exclusions
    exclude_dirs = {
        "__pycache__",
        ".pytest_cache",
        "venv",
        ".venv",
        "env",
        ".git",
        ".idea",
        ".vscode",
    }
    exclude_extensions = {".pyc", ".pyo", ".pyd"}

    print(f"Creating project submission archive: {zip_path}")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in root_dir.rglob("*"):
            # Check exclusions
            if any(part in exclude_dirs for part in file_path.parts):
                continue
            if file_path.suffix in exclude_extensions:
                continue
            if file_path.name.endswith(".zip"):
                continue

            rel_path = file_path.relative_to(root_dir)
            arcname = Path("SmartTrack-AI") / rel_path
            if file_path.is_file():
                zipf.write(file_path, arcname)
                file_count += 1

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Archive created successfully: {file_count} files, {zip_size_mb:.2f} MB")
    return str(zip_path)


if __name__ == "__main__":
    package_project()
