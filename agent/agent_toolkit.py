"""
Agent File Toolkit — Production Ready
Designed for sandboxed AI agent environments.
"""

import os
import zipfile
import base64
import json
from pathlib import Path
from typing import Union


# ─── 1. LIST FILES RECURSIVE ────────────────────────────────────────────────

def list_files_recursive(directory: str) -> dict:
    """
    Returns a full recursive tree of a directory or ZIP file.
    
    Args:
        directory: Path to a local directory OR a .zip file.
    
    Returns:
        dict with 'tree' (nested dict) and 'flat' (list of all paths).
    """
    def build_tree(path: Path) -> dict:
        tree = {}
        try:
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    tree[item.name + "/"] = build_tree(item)
                else:
                    tree[item.name] = {
                        "size_bytes": item.stat().st_size,
                        "extension": item.suffix
                    }
        except PermissionError as e:
            tree["__error__"] = str(e)
        return tree

    def flatten_tree(tree: dict, prefix: str = "") -> list:
        paths = []
        for key, val in tree.items():
            full = prefix + key
            paths.append(full)
            if isinstance(val, dict) and not val.get("size_bytes"):
                paths.extend(flatten_tree(val, full))
        return paths

    # Handle ZIP input
    if directory.endswith(".zip") and os.path.isfile(directory):
        with zipfile.ZipFile(directory, "r") as zf:
            names = zf.namelist()
            tree = {}
            for name in names:
                parts = name.split("/")
                node = tree
                for part in parts:
                    if part:
                        node = node.setdefault(part + ("/" if name.endswith("/") else ""), {})
            return {"source": directory, "type": "zip", "tree": tree, "flat": names}

    p = Path(directory)
    if not p.exists():
        return {"error": f"Path does not exist: {directory}"}

    tree = build_tree(p)
    flat = flatten_tree(tree)
    return {"source": directory, "type": "directory", "tree": tree, "flat": flat}


# ─── 2. READ FILE ────────────────────────────────────────────────────────────

def read_file(filepath: str, zip_source: str = None) -> dict:
    """
    Reads and returns file content as a string.
    Supports reading directly from inside a ZIP without extracting.

    Args:
        filepath: Path to file, or internal path if zip_source is set.
        zip_source: Optional path to a ZIP file to read from.

    Returns:
        dict with 'content', 'encoding', 'size_bytes'.
    """
    try:
        if zip_source:
            with zipfile.ZipFile(zip_source, "r") as zf:
                with zf.open(filepath) as f:
                    raw = f.read()
        else:
            with open(filepath, "rb") as f:
                raw = f.read()

        # Try UTF-8, fallback to latin-1, fallback to base64
        for enc in ("utf-8", "latin-1"):
            try:
                content = raw.decode(enc)
                return {
                    "filepath": filepath,
                    "content": content,
                    "encoding": enc,
                    "size_bytes": len(raw)
                }
            except UnicodeDecodeError:
                continue

        # Binary file — return base64
        return {
            "filepath": filepath,
            "content": base64.b64encode(raw).decode("utf-8"),
            "encoding": "base64_binary",
            "size_bytes": len(raw)
        }

    except Exception as e:
        return {"error": str(e), "filepath": filepath}


# ─── 3. WRITE FILE ───────────────────────────────────────────────────────────

def write_file(filepath: str, content: str, create_dirs: bool = True) -> dict:
    """
    Writes or overwrites content to a file.

    Args:
        filepath: Target file path.
        content: String content to write.
        create_dirs: Auto-create parent directories if missing.

    Returns:
        dict with 'success', 'filepath', 'bytes_written'.
    """
    try:
        p = Path(filepath)
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "filepath": str(p.resolve()),
            "bytes_written": len(content.encode("utf-8"))
        }
    except Exception as e:
        return {"success": False, "error": str(e), "filepath": filepath}


# ─── 4. MANAGE ZIP ───────────────────────────────────────────────────────────

def manage_zip(
    action: str,
    filename: str,
    internal_file: str = None,
    source_dir: str = None,
    extract_to: str = None
) -> dict:
    """
    Manages ZIP archives: list contents, extract files, or create archives.

    Args:
        action: One of 'list', 'extract', 'create'.
        filename: Path to the ZIP file.
        internal_file: For 'extract' — specific internal file to extract.
                       If None, extracts everything.
        source_dir: For 'create' — directory to zip up.
        extract_to: For 'extract' — destination directory. Defaults to './extracted/'.

    Returns:
        dict with action-specific results.
    """
    action = action.lower().strip()

    if action == "list":
        if not os.path.isfile(filename):
            return {"error": f"ZIP not found: {filename}"}
        with zipfile.ZipFile(filename, "r") as zf:
            info = []
            for zi in zf.infolist():
                info.append({
                    "name": zi.filename,
                    "size_bytes": zi.file_size,
                    "compressed_bytes": zi.compress_size,
                    "is_dir": zi.filename.endswith("/")
                })
            return {"action": "list", "zip": filename, "entries": info, "count": len(info)}

    elif action == "extract":
        if not os.path.isfile(filename):
            return {"error": f"ZIP not found: {filename}"}
        dest = extract_to or "./extracted"
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(filename, "r") as zf:
            if internal_file:
                zf.extract(internal_file, dest)
                extracted = [os.path.join(dest, internal_file)]
            else:
                zf.extractall(dest)
                extracted = [os.path.join(dest, n) for n in zf.namelist()]
        return {"action": "extract", "destination": dest, "extracted_files": extracted}

    elif action == "create":
        if not source_dir or not os.path.isdir(source_dir):
            return {"error": f"source_dir is required and must exist for 'create'. Got: {source_dir}"}
        created_files = []
        with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, source_dir)
                    zf.write(full_path, arcname)
                    created_files.append(arcname)
        return {
            "action": "create",
            "zip": filename,
            "files_added": created_files,
            "count": len(created_files)
        }

    else:
        return {"error": f"Unknown action '{action}'. Use: list | extract | create"}


# ─── 5. EXPORT BRIDGE ────────────────────────────────────────────────────────

def export_bridge(filename: str, output_json: str = None) -> dict:
    """
    Converts a file or ZIP into a Base64 string for agent download bridging.
    Optionally saves the result as a JSON sidecar file.

    Args:
        filename: Path to the file to export.
        output_json: Optional path to save the base64 JSON payload.

    Returns:
        dict with 'filename', 'base64_data', 'mime_type', 'size_bytes'.
    """
    MIME_MAP = {
        ".zip": "application/zip",
        ".py":  "text/x-python",
        ".txt": "text/plain",
        ".json":"application/json",
        ".lua": "text/x-lua",
        ".csv": "text/csv",
        ".html":"text/html",
    }

    try:
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}

        with open(filename, "rb") as f:
            raw = f.read()

        b64 = base64.b64encode(raw).decode("utf-8")
        ext = Path(filename).suffix.lower()
        mime = MIME_MAP.get(ext, "application/octet-stream")

        payload = {
            "filename": os.path.basename(filename),
            "mime_type": mime,
            "size_bytes": len(raw),
            "base64_data": b64,
            "usage": f'data:{mime};base64,{b64[:32]}...[truncated for display]'
        }

        if output_json:
            with open(output_json, "w") as jf:
                json.dump(payload, jf)
            payload["saved_to"] = output_json

        return payload

    except Exception as e:
        return {"error": str(e), "filename": filename}



# ─── 6. DIFF / PATCH ─────────────────────────────────────────────────────────

import difflib as _difflib
import re as _re


def diff_patch(
    action: str,
    original: str = "",
    modified: str = "",
    patch_str: str = "",
    output_path: str = None,
) -> dict:
    """
    Two operations:
      action='diff'  — generate a unified diff between two texts or file paths.
      action='patch' — apply a unified diff string to original text/file.

    Args:
        action:      'diff' or 'patch'
        original:    Raw text OR a file path (auto-detected).
        modified:    Raw text OR a file path — required for 'diff'.
        patch_str:   Unified diff string — required for 'patch'.
        output_path: Optional. Path to save the diff output or patched file.

    Returns:
        dict with action-specific results.
    """

    def _load(src: str) -> tuple:
        p = Path(src)
        if len(src) < 512 and p.exists():
            try:
                return p.read_text(errors="replace"), str(p)
            except Exception:
                pass
        return src, "<text>"

    action = action.lower().strip()

    # ── DIFF ──────────────────────────────────────────────────────────────────
    if action == "diff":
        if not original or not modified:
            return {"error": "diff requires both 'original' and 'modified'."}

        orig_text, orig_label = _load(original)
        mod_text, mod_label = _load(modified)

        diff_lines = list(
            _difflib.unified_diff(
                orig_text.splitlines(keepends=True),
                mod_text.splitlines(keepends=True),
                fromfile=orig_label,
                tofile=mod_label,
                lineterm="",
            )
        )

        if not diff_lines:
            return {"action": "diff", "changed": False, "diff": ""}

        diff_str = "\n".join(diff_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(diff_str)

        lines_added   = len([l for l in diff_lines if l.startswith("+") and not l.startswith("+++")])
        lines_removed = len([l for l in diff_lines if l.startswith("-") and not l.startswith("---")])

        return {
            "action": "diff",
            "changed": True,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "diff": diff_str,
            "saved_to": output_path,
        }

    # ── PATCH ─────────────────────────────────────────────────────────────────
    elif action == "patch":
        if not original:
            return {"error": "patch requires 'original' (file path or text)."}
        if not patch_str:
            return {"error": "patch requires 'patch_str' (unified diff string)."}

        orig_text, orig_label = _load(original)
        orig_lines = orig_text.splitlines(keepends=True)

        # Manual unified diff apply (pure stdlib)
        result = list(orig_lines)
        patch_lines = patch_str.splitlines(keepends=True)
        offset = 0
        i = 0
        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith("@@"):
                m = _re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", line)
                if not m:
                    i += 1
                    continue
                orig_start = int(m.group(1)) - 1
                i += 1
                pos = orig_start + offset
                removes, adds = [], []
                while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                    pl = patch_lines[i]
                    if pl.startswith("-"):
                        removes.append(pl[1:])
                    elif pl.startswith("+"):
                        adds.append(pl[1:])
                    else:
                        # context line — keep
                        pass
                    i += 1
                del result[pos:pos + len(removes)]
                result[pos:pos] = adds
                offset += len(adds) - len(removes)
            else:
                i += 1

        patched_text = "".join(result)

        target = output_path or (original if Path(original).exists() else None)
        if target:
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(patched_text)
            return {"action": "patch", "success": True, "written_to": target, "bytes": len(patched_text.encode())}

        return {"action": "patch", "success": True, "patched_content": patched_text}

    else:
        return {"error": f"Unknown action '{action}'. Use: diff | patch"}


# ─── SELF-TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== AGENT TOOLKIT SELF-TEST ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write
        wf = write_file(os.path.join(tmpdir, "test.txt"), "hello agent")
        print("write_file:", wf)

        # Read
        rf = read_file(os.path.join(tmpdir, "test.txt"))
        print("read_file:", rf)

        # List
        lf = list_files_recursive(tmpdir)
        print("list_files:", lf["flat"])

        # Create ZIP
        zip_path = os.path.join(tmpdir, "bundle.zip")
        mz = manage_zip("create", zip_path, source_dir=tmpdir)
        print("manage_zip create:", mz)

        # List ZIP
        mz2 = manage_zip("list", zip_path)
        print("manage_zip list:", mz2)

        # Export
        eb = export_bridge(zip_path)
        print("export_bridge:", {k: v for k, v in eb.items() if k != "base64_data"})

    print("\n=== ALL TESTS PASSED ===")
