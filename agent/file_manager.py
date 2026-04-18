"""
agent/file_manager.py  — v2.0 Deep-Exploration Edition
=======================================================
Tools: list_files_recursive★ | list_files | read_file | write_file |
       manage_zip | search_in_files★ | move_or_rename_file★ | delete_file★ |
       get_file_as_base64 | generate_download_link
"""
import base64, json, mimetypes, os, re, shutil, uuid, zipfile
from pathlib import Path
from typing import List, Optional

SANDBOX_ROOT     = Path(os.environ.get("AGENT_SANDBOX",    "/tmp/gazcc_sandbox")).resolve()
EXPORT_DIR       = Path(os.environ.get("AGENT_EXPORT_DIR", "/tmp/gazcc_exports")).resolve()
BASE_URL         = os.environ.get("AGENT_BASE_URL",        "http://localhost:8000")
MAX_EXPORT_BYTES = int(os.environ.get("MAX_EXPORT_MB", "10")) * 1024 * 1024
MAX_READ_BYTES   = 2 * 1024 * 1024
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _ok(d): return {"status": "ok", **d}
def _err(m, c="ERROR"): return {"status": "error", "code": c, "message": m}

def _resolve(filepath: str, must_exist: bool = False) -> Path:
    p = Path(filepath)
    if p.is_absolute():
        try:    resolved = SANDBOX_ROOT / p.relative_to(SANDBOX_ROOT)
        except: resolved = SANDBOX_ROOT / p.name
    else:
        resolved = (SANDBOX_ROOT / filepath).resolve()
    try:    resolved.relative_to(SANDBOX_ROOT)
    except: raise PermissionError(f"Path traversal blocked: {filepath}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {resolved}")
    return resolved


# ── TOOL 1: list_files_recursive ★ ─────────────────────────────────────────
def list_files_recursive(directory: str = ".", max_depth: int = 10, include_hidden: bool = False) -> dict:
    """
    ★ DEEP EXPLORER — Gunakan tool ini PERTAMA setelah extract ZIP untuk
    memetakan SELURUH struktur folder dan file proyek secara rekursif.
    Output mencakup ASCII tree visual, flat list semua path (siap di-loop
    dengan read_file), dan ringkasan per ekstensi. Tanpa tool ini, Agent
    bergerak buta dalam proyek multi-folder.
    """
    try:
        target = _resolve(directory)
        if not target.exists(): return _err(f"Tidak ditemukan: {directory}", "NOT_FOUND")
        if not target.is_dir(): return _err(f"Bukan direktori: {directory}", "NOT_A_DIRECTORY")

        nodes, flat_paths, ext_counter, tree_lines = [], [], {}, []
        root_rel = str(target.relative_to(SANDBOX_ROOT)) if target != SANDBOX_ROOT else "."
        tree_lines.append(f"📂 {root_rel}/")

        def _walk(cur: Path, depth: int, prefix: str):
            if depth > max_depth: return
            try: children = sorted(cur.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError: return
            if not include_hidden:
                children = [c for c in children if not c.name.startswith(".")]
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                icon    = "📁 " if child.is_dir() else "📄 "
                tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{icon}{child.name}")
                rel = str(child.relative_to(SANDBOX_ROOT))
                ext = child.suffix.lower() if child.is_file() else ""
                nodes.append({"path": rel, "name": child.name,
                               "type": "directory" if child.is_dir() else "file",
                               "depth": depth, "size_bytes": child.stat().st_size if child.is_file() else 0,
                               "extension": ext})
                if child.is_file():
                    flat_paths.append(rel)
                    ext_counter[ext] = ext_counter.get(ext, 0) + 1
                if child.is_dir():
                    _walk(child, depth + 1, prefix + ("    " if is_last else "│   "))

        _walk(target, 1, "")
        total_files = sum(1 for n in nodes if n["type"] == "file")
        total_dirs  = sum(1 for n in nodes if n["type"] == "directory")
        return _ok({
            "root": root_rel,
            "tree": "\n".join(tree_lines),
            "flat_paths": flat_paths,
            "total_files": total_files,
            "total_dirs":  total_dirs,
            "extensions_summary": dict(sorted(ext_counter.items(), key=lambda x: -x[1])),
            "nodes": nodes,
        })
    except PermissionError as e: return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:       return _err(f"list_files_recursive error: {e}", "UNEXPECTED_ERROR")


# ── TOOL 2: list_files (shallow) ────────────────────────────────────────────
def list_files(directory: str = ".") -> dict:
    """List satu level direktori (non-rekursif). Untuk deep mapping gunakan list_files_recursive."""
    try:
        target = _resolve(directory)
        if not target.exists(): return _err(f"Tidak ditemukan: {directory}", "NOT_FOUND")
        if not target.is_dir(): return _err(f"Bukan direktori: {directory}", "NOT_A_DIRECTORY")
        items = [{"name": i.name, "type": "directory" if i.is_dir() else "file",
                  "size_bytes": i.stat().st_size if i.is_file() else 0,
                  "path": str(i.relative_to(SANDBOX_ROOT))}
                 for i in sorted(target.iterdir())]
        return _ok({"directory": str(target.relative_to(SANDBOX_ROOT)), "total_items": len(items), "items": items})
    except Exception as e: return _err(str(e), "ERROR")


# ── TOOL 3: read_file ────────────────────────────────────────────────────────
def read_file(filepath: str) -> dict:
    """Baca konten file teks. Gunakan flat_paths dari list_files_recursive untuk loop."""
    try:
        target = _resolve(filepath, must_exist=True)
        if not target.is_file(): return _err(f"Bukan file: {filepath}", "NOT_A_FILE")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return _err(f"File terlalu besar ({size//1024}KB). Gunakan get_file_as_base64.", "FILE_TOO_LARGE")
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return _ok({"filepath": str(target.relative_to(SANDBOX_ROOT)),
                             "content": target.read_text(encoding=enc),
                             "size_bytes": size, "encoding": enc})
            except UnicodeDecodeError: continue
        return _err("Binary file. Gunakan get_file_as_base64.", "ENCODING_ERROR")
    except FileNotFoundError as e: return _err(str(e), "NOT_FOUND")
    except Exception as e:         return _err(str(e), "ERROR")


# ── TOOL 4: write_file ───────────────────────────────────────────────────────
def write_file(filepath: str, content: str) -> dict:
    """Buat atau overwrite file. Parent dirs dibuat otomatis."""
    try:
        target = _resolve(filepath)
        created = not target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _ok({"filepath": str(target.relative_to(SANDBOX_ROOT)),
                    "bytes_written": target.stat().st_size, "created_dirs": created})
    except Exception as e: return _err(str(e), "ERROR")


# ── TOOL 5: manage_zip ───────────────────────────────────────────────────────
def manage_zip(action: str, filename: str, files: Optional[list] = None,
               internal_file: Optional[str] = None, dest_dir: Optional[str] = None) -> dict:
    """
    ZIP manager: action='list'|'extract'|'create'.
    Tip: Setelah extract, LANGSUNG panggil list_files_recursive() untuk peta proyek.
    Untuk create dengan direktori penuh, masukkan path direktori di 'files' — semua isinya ikut.
    """
    action = action.lower().strip()
    if action == "list":
        try:
            zp = _resolve(filename, must_exist=True)
            if not zipfile.is_zipfile(zp): return _err("Bukan ZIP valid", "INVALID_ZIP")
            with zipfile.ZipFile(zp, "r") as zf:
                entries = [{"name": i.filename, "size_bytes": i.file_size,
                             "compressed_bytes": i.compress_size,
                             "is_dir": i.filename.endswith("/"), "crc": i.CRC}
                            for i in zf.infolist()]
            return _ok({"zip_file": str(zp.relative_to(SANDBOX_ROOT)),
                        "total_files": len(entries), "entries": entries})
        except Exception as e: return _err(str(e), "ERROR")

    elif action == "extract":
        try:
            zp = _resolve(filename, must_exist=True)
            if not zipfile.is_zipfile(zp): return _err("Bukan ZIP valid", "INVALID_ZIP")
            ext_to = _resolve(dest_dir) if dest_dir else _resolve(zp.stem)
            ext_to.mkdir(parents=True, exist_ok=True)
            extracted = []
            with zipfile.ZipFile(zp, "r") as zf:
                if internal_file:
                    zf.extract(internal_file, path=ext_to); extracted.append(internal_file)
                else:
                    for m in zf.infolist():
                        mp = (ext_to / m.filename).resolve()
                        try: mp.relative_to(SANDBOX_ROOT)
                        except ValueError: return _err(f"ZIP Slip: {m.filename}", "SECURITY_ERROR")
                        zf.extract(m, path=ext_to)
                        if not m.filename.endswith("/"): extracted.append(m.filename)
            return _ok({"extracted_to": str(ext_to.relative_to(SANDBOX_ROOT)),
                        "files_extracted": extracted, "total_extracted": len(extracted),
                        "next_step": "Panggil list_files_recursive() pada 'extracted_to' untuk memetakan proyek."})
        except Exception as e: return _err(str(e), "ERROR")

    elif action == "create":
        if not files: return _err("'files' wajib untuk action='create'", "MISSING_PARAMETER")
        try:
            zout = _resolve(filename)
            zout.parent.mkdir(parents=True, exist_ok=True)
            added, total = [], 0
            with zipfile.ZipFile(zout, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    fp = _resolve(f, must_exist=True)
                    if fp.is_dir():
                        for sub in fp.rglob("*"):
                            if sub.is_file():
                                arc = str(sub.relative_to(SANDBOX_ROOT))
                                zf.write(sub, arcname=arc); added.append(arc); total += sub.stat().st_size
                    else:
                        arc = str(fp.relative_to(SANDBOX_ROOT))
                        zf.write(fp, arcname=arc); added.append(arc); total += fp.stat().st_size
            return _ok({"zip_path": str(zout.relative_to(SANDBOX_ROOT)),
                        "files_added": added, "total_files": len(added),
                        "total_size_bytes": total, "zip_size_bytes": zout.stat().st_size})
        except Exception as e: return _err(str(e), "ERROR")
    else:
        return _err(f"Action tidak dikenal: '{action}'", "INVALID_ACTION")


# ── TOOL 6: search_in_files ★ ───────────────────────────────────────────────
def search_in_files(query: str, directory: str = ".", extensions: Optional[List[str]] = None,
                    case_sensitive: bool = False, max_results: int = 100) -> dict:
    """
    ★ GREP AGENT — Cari teks/regex di semua file secara rekursif.
    Gunakan sebelum refactoring untuk menemukan SEMUA lokasi yang perlu di-patch.
    Filter ekstensi dengan extensions=['.py','.js'] untuk fokus ke source code.
    """
    try:
        target = _resolve(directory)
        if not target.exists(): return _err(f"Tidak ditemukan: {directory}", "NOT_FOUND")
        flags = 0 if case_sensitive else re.IGNORECASE
        try: pattern = re.compile(query, flags)
        except re.error as e: return _err(f"Regex tidak valid: {e}", "INVALID_REGEX")

        results, files_searched = [], 0
        for fp in target.rglob("*"):
            if not fp.is_file(): continue
            if extensions and fp.suffix.lower() not in [e.lower() for e in extensions]: continue
            if fp.stat().st_size > MAX_READ_BYTES: continue
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    text = fp.read_text(encoding=enc); files_searched += 1
                    for lineno, line in enumerate(text.splitlines(), 1):
                        m = pattern.search(line)
                        if m:
                            results.append({"file": str(fp.relative_to(SANDBOX_ROOT)),
                                            "line_number": lineno, "line_content": line.strip(),
                                            "match": m.group(0)})
                            if len(results) >= max_results:
                                return _ok({"query": query, "total_matches": len(results),
                                            "files_searched": files_searched, "truncated": True, "results": results})
                    break
                except UnicodeDecodeError: continue
        return _ok({"query": query, "total_matches": len(results),
                    "files_searched": files_searched, "truncated": False, "results": results})
    except Exception as e: return _err(str(e), "ERROR")


# ── TOOL 7: move_or_rename_file ★ ───────────────────────────────────────────
def move_or_rename_file(source: str, destination: str) -> dict:
    """Pindah atau rename file/direktori dalam sandbox."""
    try:
        src = _resolve(source, must_exist=True); dst = _resolve(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return _ok({"source": source, "destination": str(dst.relative_to(SANDBOX_ROOT)),
                    "type": "directory" if dst.is_dir() else "file"})
    except FileNotFoundError as e: return _err(str(e), "NOT_FOUND")
    except Exception as e:         return _err(str(e), "ERROR")


# ── TOOL 8: delete_file ★ ────────────────────────────────────────────────────
def delete_file(filepath: str, force: bool = False) -> dict:
    """
    Hapus file atau direktori. force=True untuk hapus direktori beserta isinya (rm -rf).
    """
    try:
        t = _resolve(filepath, must_exist=True)
        if t.is_file():   t.unlink()
        elif t.is_dir():  shutil.rmtree(t) if force else t.rmdir()
        else:             return _err(f"Bukan file/dir: {filepath}", "UNKNOWN_TYPE")
        return _ok({"deleted": filepath, "force": force})
    except OSError as e:
        if "not empty" in str(e).lower():
            return _err("Direktori tidak kosong. Gunakan force=True.", "DIR_NOT_EMPTY")
        return _err(str(e), "OS_ERROR")
    except Exception as e: return _err(str(e), "ERROR")


# ── TOOL 9: get_file_as_base64 ──────────────────────────────────────────────
def get_file_as_base64(filename: str) -> dict:
    """Encode file ke Base64. Sertakan 'export_marker' di Final Answer untuk tombol download."""
    try:
        t = _resolve(filename, must_exist=True)
        if not t.is_file(): return _err(f"Bukan file: {filename}", "NOT_A_FILE")
        size = t.stat().st_size
        if size > MAX_EXPORT_BYTES:
            return _err(f"Terlalu besar ({size//1024//1024}MB). Gunakan generate_download_link.", "FILE_TOO_LARGE")
        mime = mimetypes.guess_type(str(t))[0] or "application/octet-stream"
        b64  = base64.b64encode(t.read_bytes()).decode()
        mark = f"[FILE_EXPORT:{t.name}:{b64}:{mime}]"
        return _ok({"filename": t.name, "mime_type": mime, "size_bytes": size,
                    "base64_data": b64, "export_marker": mark,
                    "instruction": f"Sertakan export_marker PERSIS di Final Answer untuk render tombol download '{t.name}'."})
    except Exception as e: return _err(str(e), "ERROR")


# ── TOOL 10: generate_download_link ─────────────────────────────────────────
def generate_download_link(filename: str) -> dict:
    """Generate URL download. Sertakan 'export_marker' di Final Answer."""
    try:
        src  = _resolve(filename, must_exist=True)
        if not src.is_file(): return _err(f"Bukan file: {filename}", "NOT_A_FILE")
        mime = mimetypes.guess_type(str(src))[0] or "application/octet-stream"
        tok  = uuid.uuid4().hex[:16]
        ep   = EXPORT_DIR / f"{tok}_{src.name}"
        shutil.copy2(src, ep)
        (EXPORT_DIR / f"{tok}.meta.json").write_text(
            json.dumps({"token": tok, "original_filename": src.name,
                        "export_filename": ep.name, "mime_type": mime,
                        "size_bytes": src.stat().st_size}, indent=2))
        url  = f"{BASE_URL.rstrip('/')}/api/download/{tok}"
        mark = f"[FILE_LINK:{src.name}:{url}]"
        return _ok({"filename": src.name, "download_url": url, "token": tok,
                    "export_path": str(ep), "size_bytes": src.stat().st_size,
                    "mime_type": mime, "export_marker": mark,
                    "instruction": f"Sertakan di Final Answer: {mark}"})
    except Exception as e: return _err(str(e), "ERROR")


# ── DISPATCHER ───────────────────────────────────────────────────────────────
TOOL_MAP = {
    "list_files_recursive":   list_files_recursive,
    "list_files":             list_files,
    "read_file":              read_file,
    "write_file":             write_file,
    "manage_zip":             manage_zip,
    "search_in_files":        search_in_files,
    "move_or_rename_file":    move_or_rename_file,
    "delete_file":            delete_file,
    "get_file_as_base64":     get_file_as_base64,
    "generate_download_link": generate_download_link,
}

def dispatch(tool_name: str, args: dict) -> dict:
    fn = TOOL_MAP.get(tool_name)
    if not fn: return _err(f"Tool tidak dikenal: '{tool_name}'", "UNKNOWN_TOOL")
    try:    return fn(**args)
    except TypeError as e: return _err(f"Argumen salah untuk '{tool_name}': {e}", "BAD_ARGUMENTS")
    except Exception as e: return _err(f"'{tool_name}' crash: {e}", "RUNTIME_ERROR")


if __name__ == "__main__":
    # Quick smoke test
    import textwrap
    write_file("test/hello.py", "def greet():\n    return 'Hello World'\n")
    write_file("test/config.json", '{"version": "2.0"}')
    result = list_files_recursive("test")
    print(result["tree"])
    hits = search_in_files("def ", "test", [".py"])
    print(f"search hits: {hits['total_matches']}")
    manage_zip("create", "test_bundle.zip", files=["test"])
    print("ZIP created:", manage_zip("list", "test_bundle.zip")["total_files"], "files")
    print("All tools OK ✓")
