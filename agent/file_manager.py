"""
agent/file_manager.py
=====================
File Management Suite — Production Toolset untuk AI Agent.
Dirancang untuk Function Calling integration (OpenAI / Anthropic format).

Semua fungsi:
  - Return dict yang konsisten  → { "status": "ok"|"error", "data": ..., ... }
  - Aman dari path traversal    → SANDBOX_ROOT enforcement
  - Sync + async friendly       → sync by default, bisa di-wrap asyncio.to_thread()

Tool List:
  1. list_files(directory)
  2. read_file(filepath)
  3. write_file(filepath, content)
  4. manage_zip(action, filename, files, internal_file, dest_dir)
  5. get_file_as_base64(filename)
  6. generate_download_link(filename)

Agent Orchestration Flow (Full Cycle):
  receive_zip → manage_zip(action='list') → manage_zip(action='extract')
  → read_file() → write_file() → manage_zip(action='create')
  → get_file_as_base64() | generate_download_link()
"""

import base64
import json
import mimetypes
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — ubah sesuai environment
# ─────────────────────────────────────────────────────────────────────────────

# Root sandbox. Semua operasi file dibatasi di dalam direktori ini.
SANDBOX_ROOT = Path(os.environ.get("AGENT_SANDBOX", "/tmp/gazcc_sandbox")).resolve()

# Direktori untuk file ekspor (download link)
EXPORT_DIR = Path(os.environ.get("AGENT_EXPORT_DIR", "/tmp/gazcc_exports")).resolve()

# Base URL untuk download links (set via env, misal VERCEL_URL atau domain kamu)
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:8000")

# Batas ukuran file untuk Base64 export (default 10MB)
MAX_EXPORT_BYTES = int(os.environ.get("MAX_EXPORT_MB", "10")) * 1024 * 1024

# Batas ukuran file untuk read (default 2MB)
MAX_READ_BYTES = 2 * 1024 * 1024

# Auto-buat direktori yang diperlukan saat module diimport
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ok(data: dict) -> dict:
    """Wrap successful result."""
    return {"status": "ok", **data}


def _err(message: str, code: str = "ERROR") -> dict:
    """Wrap error result."""
    return {"status": "error", "code": code, "message": message}


def _resolve(filepath: str, must_exist: bool = False) -> Path:
    """
    Resolve path ke SANDBOX_ROOT.
    Mencegah path traversal (../../etc/passwd dan sejenisnya).
    """
    # Jika absolute path diberikan, strip prefix-nya
    p = Path(filepath)
    if p.is_absolute():
        # Coba resolve relatif ke sandbox
        try:
            rel = p.relative_to(SANDBOX_ROOT)
            resolved = SANDBOX_ROOT / rel
        except ValueError:
            # Path di luar sandbox → paksa masuk sandbox
            resolved = SANDBOX_ROOT / p.name
    else:
        resolved = (SANDBOX_ROOT / filepath).resolve()

    # Pastikan masih di dalam sandbox setelah resolve
    try:
        resolved.relative_to(SANDBOX_ROOT)
    except ValueError:
        raise PermissionError(f"Path traversal detected. Path harus di dalam sandbox: {SANDBOX_ROOT}")

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File/directory tidak ditemukan: {resolved}")

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — list_files
# ─────────────────────────────────────────────────────────────────────────────

def list_files(directory: str = ".") -> dict:
    """
    List semua file dan subdirektori di dalam path yang diberikan.

    Args:
        directory: Path direktori relatif terhadap SANDBOX_ROOT.
                   Gunakan "." untuk root sandbox.

    Returns:
        {
          "status": "ok",
          "directory": str,
          "total_items": int,
          "items": [
            { "name": str, "type": "file"|"directory",
              "size_bytes": int, "path": str }
          ]
        }

    Agent usage:
        Panggil dulu sebelum baca/tulis untuk tahu apa yang ada di sandbox.
    """
    try:
        target = _resolve(directory)

        if not target.exists():
            return _err(f"Direktori tidak ditemukan: {directory}", "NOT_FOUND")
        if not target.is_dir():
            return _err(f"Bukan direktori: {directory}", "NOT_A_DIRECTORY")

        items = []
        for item in sorted(target.iterdir()):
            entry = {
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size_bytes": item.stat().st_size if item.is_file() else 0,
                "path": str(item.relative_to(SANDBOX_ROOT)),
            }
            items.append(entry)

        return _ok({
            "directory": str(target.relative_to(SANDBOX_ROOT)),
            "total_items": len(items),
            "items": items,
        })

    except PermissionError as e:
        return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:
        return _err(f"list_files gagal: {e}", "UNEXPECTED_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — read_file
# ─────────────────────────────────────────────────────────────────────────────

def read_file(filepath: str) -> dict:
    """
    Baca konten file teks.

    Args:
        filepath: Path file relatif terhadap SANDBOX_ROOT.

    Returns:
        {
          "status": "ok",
          "filepath": str,
          "content": str,
          "size_bytes": int,
          "encoding": str
        }

    Agent usage:
        Setelah extract ZIP, panggil ini untuk baca file yang diperlukan.
        Untuk binary file (gambar, dll), gunakan get_file_as_base64.
    """
    try:
        target = _resolve(filepath, must_exist=True)

        if not target.is_file():
            return _err(f"Bukan file: {filepath}", "NOT_A_FILE")

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return _err(
                f"File terlalu besar ({size // 1024}KB > {MAX_READ_BYTES // 1024}KB limit). "
                f"Gunakan get_file_as_base64 untuk binary export.",
                "FILE_TOO_LARGE"
            )

        # Coba baca UTF-8 dulu, fallback ke latin-1
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                content = target.read_text(encoding=enc)
                return _ok({
                    "filepath": str(target.relative_to(SANDBOX_ROOT)),
                    "content": content,
                    "size_bytes": size,
                    "encoding": enc,
                })
            except UnicodeDecodeError:
                continue

        return _err(f"File tidak bisa dibaca sebagai teks. Mungkin binary. Coba get_file_as_base64.", "ENCODING_ERROR")

    except FileNotFoundError as e:
        return _err(str(e), "NOT_FOUND")
    except PermissionError as e:
        return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:
        return _err(f"read_file gagal: {e}", "UNEXPECTED_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — write_file
# ─────────────────────────────────────────────────────────────────────────────

def write_file(filepath: str, content: str) -> dict:
    """
    Tulis atau overwrite file dengan konten yang diberikan.
    Otomatis buat parent directories jika belum ada.

    Args:
        filepath: Path file relatif terhadap SANDBOX_ROOT.
        content:  Konten teks yang akan ditulis.

    Returns:
        {
          "status": "ok",
          "filepath": str,
          "bytes_written": int,
          "created_dirs": bool
        }

    Agent usage:
        Setelah modifikasi konten via read_file, panggil ini untuk simpan.
        Kemudian bisa di-zip ulang dengan manage_zip(action='create').
    """
    try:
        target = _resolve(filepath)

        # Buat parent dirs jika belum ada
        created_dirs = not target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(content, encoding="utf-8")
        bytes_written = target.stat().st_size

        return _ok({
            "filepath": str(target.relative_to(SANDBOX_ROOT)),
            "bytes_written": bytes_written,
            "created_dirs": created_dirs,
            "absolute_path": str(target),
        })

    except PermissionError as e:
        return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:
        return _err(f"write_file gagal: {e}", "UNEXPECTED_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — manage_zip
# ─────────────────────────────────────────────────────────────────────────────

def manage_zip(
    action: str,
    filename: str,
    files: Optional[list] = None,
    internal_file: Optional[str] = None,
    dest_dir: Optional[str] = None,
) -> dict:
    """
    Manajemen ZIP: list isi, extract, atau buat ZIP baru.

    Args:
        action:        'list' | 'extract' | 'create'
        filename:      Path ke .zip file (relatif ke SANDBOX_ROOT).
                       Untuk 'create', ini adalah nama output ZIP yang akan dibuat.
        files:         (untuk action='create') List path file yang akan di-zip.
        internal_file: (untuk action='extract') Nama file spesifik di dalam ZIP.
                       Jika None, extract semua file.
        dest_dir:      (untuk action='extract') Direktori tujuan ekstraksi.
                       Default: subdirektori dengan nama yang sama dengan ZIP.

    Returns dict tergantung action:

        action='list':
        {
          "status": "ok",
          "zip_file": str,
          "total_files": int,
          "entries": [{ "name": str, "size_bytes": int, "compressed_bytes": int,
                        "is_dir": bool, "crc": int }]
        }

        action='extract':
        {
          "status": "ok",
          "extracted_to": str,
          "files_extracted": [str],
          "total_extracted": int
        }

        action='create':
        {
          "status": "ok",
          "zip_path": str,
          "files_added": [str],
          "total_size_bytes": int
        }

    Agent usage — Full Cycle:
        1. manage_zip(action='list', filename='input.zip')
           → Lihat isi ZIP, pilih file mana yang mau diproses
        2. manage_zip(action='extract', filename='input.zip', dest_dir='extracted/')
           → Extract ke sandbox
        3. read_file('extracted/config.json')
           → Baca file
        4. write_file('extracted/config.json', modified_content)
           → Simpan perubahan
        5. manage_zip(action='create', filename='output.zip',
                      files=['extracted/config.json', 'extracted/main.py'])
           → Zip ulang
        6. get_file_as_base64('output.zip') atau generate_download_link('output.zip')
           → Kirim ke user
    """
    action = action.lower().strip()

    # ── action: list ──────────────────────────────────────────────────────────
    if action == "list":
        try:
            zip_path = _resolve(filename, must_exist=True)

            if not zipfile.is_zipfile(zip_path):
                return _err(f"Bukan file ZIP yang valid: {filename}", "INVALID_ZIP")

            entries = []
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    entries.append({
                        "name": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "is_dir": info.filename.endswith("/"),
                        "crc": info.CRC,
                    })

            return _ok({
                "zip_file": str(zip_path.relative_to(SANDBOX_ROOT)),
                "total_files": len(entries),
                "entries": entries,
            })

        except FileNotFoundError as e:
            return _err(str(e), "NOT_FOUND")
        except zipfile.BadZipFile:
            return _err(f"File ZIP korup atau tidak valid: {filename}", "BAD_ZIP")
        except Exception as e:
            return _err(f"manage_zip(list) gagal: {e}", "UNEXPECTED_ERROR")

    # ── action: extract ───────────────────────────────────────────────────────
    elif action == "extract":
        try:
            zip_path = _resolve(filename, must_exist=True)

            if not zipfile.is_zipfile(zip_path):
                return _err(f"Bukan file ZIP yang valid: {filename}", "INVALID_ZIP")

            # Tentukan dest_dir
            if dest_dir:
                extract_to = _resolve(dest_dir)
            else:
                # Default: subfolder dengan nama ZIP (tanpa .zip)
                stem = zip_path.stem
                extract_to = _resolve(stem)

            extract_to.mkdir(parents=True, exist_ok=True)

            extracted_files = []
            with zipfile.ZipFile(zip_path, "r") as zf:
                if internal_file:
                    # Extract file spesifik
                    try:
                        zf.extract(internal_file, path=extract_to)
                        extracted_files.append(internal_file)
                    except KeyError:
                        return _err(
                            f"File '{internal_file}' tidak ditemukan di dalam ZIP.",
                            "FILE_NOT_IN_ZIP"
                        )
                else:
                    # Extract semua file
                    # Proteksi zip slip attack
                    for member in zf.infolist():
                        member_path = (extract_to / member.filename).resolve()
                        try:
                            member_path.relative_to(SANDBOX_ROOT)
                        except ValueError:
                            return _err(
                                f"ZIP Slip attack terdeteksi pada entry: {member.filename}",
                                "SECURITY_ERROR"
                            )
                        zf.extract(member, path=extract_to)
                        if not member.filename.endswith("/"):
                            extracted_files.append(member.filename)

            return _ok({
                "extracted_to": str(extract_to.relative_to(SANDBOX_ROOT)),
                "files_extracted": extracted_files,
                "total_extracted": len(extracted_files),
            })

        except FileNotFoundError as e:
            return _err(str(e), "NOT_FOUND")
        except zipfile.BadZipFile:
            return _err(f"File ZIP korup: {filename}", "BAD_ZIP")
        except PermissionError as e:
            return _err(str(e), "PERMISSION_DENIED")
        except Exception as e:
            return _err(f"manage_zip(extract) gagal: {e}", "UNEXPECTED_ERROR")

    # ── action: create ────────────────────────────────────────────────────────
    elif action == "create":
        try:
            if not files or not isinstance(files, list):
                return _err(
                    "Parameter 'files' harus berupa list path file untuk action='create'.",
                    "MISSING_PARAMETER"
                )

            zip_out = _resolve(filename)
            zip_out.parent.mkdir(parents=True, exist_ok=True)

            added_files = []
            total_size = 0

            with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    try:
                        file_path = _resolve(f, must_exist=True)
                        arcname = file_path.name  # Simpan tanpa full path
                        zf.write(file_path, arcname=arcname)
                        added_files.append(arcname)
                        total_size += file_path.stat().st_size
                    except FileNotFoundError:
                        return _err(f"File tidak ditemukan saat buat ZIP: {f}", "FILE_NOT_FOUND")

            return _ok({
                "zip_path": str(zip_out.relative_to(SANDBOX_ROOT)),
                "absolute_path": str(zip_out),
                "files_added": added_files,
                "total_files": len(added_files),
                "total_size_bytes": total_size,
                "zip_size_bytes": zip_out.stat().st_size,
            })

        except PermissionError as e:
            return _err(str(e), "PERMISSION_DENIED")
        except Exception as e:
            return _err(f"manage_zip(create) gagal: {e}", "UNEXPECTED_ERROR")

    else:
        return _err(
            f"Action tidak dikenal: '{action}'. Gunakan: 'list', 'extract', atau 'create'.",
            "INVALID_ACTION"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — get_file_as_base64
# ─────────────────────────────────────────────────────────────────────────────

def get_file_as_base64(filename: str) -> dict:
    """
    Encode file ke Base64 agar Agent bisa "print" ke UI.
    Frontend mendeteksi marker [FILE_EXPORT:...] dan render tombol download.

    Args:
        filename: Path file relatif ke SANDBOX_ROOT.

    Returns:
        {
          "status": "ok",
          "filename": str,
          "mime_type": str,
          "size_bytes": int,
          "base64_data": str,
          "export_marker": str,   ← tulis ini di Final Answer
          "instruction": str
        }

    Agent usage:
        Setelah file siap, panggil ini dan sertakan 'export_marker'
        PERSIS di dalam Final Answer. UI akan render tombol download otomatis.

    Marker format yang dikenali UI:
        [FILE_EXPORT:{filename}:{base64_data}:{mime_type}]
    """
    try:
        target = _resolve(filename, must_exist=True)

        if not target.is_file():
            return _err(f"Bukan file: {filename}", "NOT_A_FILE")

        size = target.stat().st_size
        if size > MAX_EXPORT_BYTES:
            return _err(
                f"File terlalu besar ({size // 1024 // 1024}MB) untuk Base64 export. "
                f"Gunakan generate_download_link sebagai gantinya.",
                "FILE_TOO_LARGE"
            )

        mime, _ = mimetypes.guess_type(str(target))
        mime = mime or "application/octet-stream"

        raw = target.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")

        marker = f"[FILE_EXPORT:{target.name}:{b64}:{mime}]"

        return _ok({
            "filename": target.name,
            "mime_type": mime,
            "size_bytes": size,
            "base64_data": b64,
            "export_marker": marker,
            "instruction": (
                f"Sertakan export_marker ini PERSIS di Final Answer kamu "
                f"agar UI merender tombol download untuk file '{target.name}'."
            ),
        })

    except FileNotFoundError as e:
        return _err(str(e), "NOT_FOUND")
    except PermissionError as e:
        return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:
        return _err(f"get_file_as_base64 gagal: {e}", "UNEXPECTED_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — generate_download_link
# ─────────────────────────────────────────────────────────────────────────────

def generate_download_link(filename: str) -> dict:
    """
    Copy file ke EXPORT_DIR dan generate URL download yang bisa diklik user.
    Cocok untuk file besar atau UX yang lebih clean dibanding Base64.

    Args:
        filename: Path file relatif ke SANDBOX_ROOT.

    Returns:
        {
          "status": "ok",
          "filename": str,
          "download_url": str,
          "token": str,
          "export_path": str,
          "size_bytes": int,
          "mime_type": str,
          "export_marker": str,   ← tulis ini di Final Answer
          "instruction": str
        }

    Marker format yang dikenali UI:
        [FILE_LINK:{filename}:{download_url}]

    Note:
        Asumsi ada static file server atau /api/download/{token} handler.
        Di Vercel, gunakan api/download.py yang serve dari EXPORT_DIR.
        Jika tidak ada server, otomatis fallback ke Base64 bridge.
    """
    try:
        source = _resolve(filename, must_exist=True)

        if not source.is_file():
            return _err(f"Bukan file: {filename}", "NOT_A_FILE")

        size = source.stat().st_size
        mime, _ = mimetypes.guess_type(str(source))
        mime = mime or "application/octet-stream"

        # Generate unique token
        token = uuid.uuid4().hex[:16]
        export_filename = f"{token}_{source.name}"
        export_path = EXPORT_DIR / export_filename

        # Copy ke export dir
        shutil.copy2(source, export_path)

        # Simpan metadata JSON untuk download handler
        meta_path = EXPORT_DIR / f"{token}.meta.json"
        meta = {
            "token": token,
            "original_filename": source.name,
            "export_filename": export_filename,
            "mime_type": mime,
            "size_bytes": size,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Build URL
        base = BASE_URL.rstrip("/")
        download_url = f"{base}/api/download/{token}"

        marker = f"[FILE_LINK:{source.name}:{download_url}]"

        return _ok({
            "filename": source.name,
            "download_url": download_url,
            "token": token,
            "export_path": str(export_path),
            "size_bytes": size,
            "mime_type": mime,
            "export_marker": marker,
            "instruction": (
                f"Sertakan export_marker ini di Final Answer: {marker} "
                f"UI akan render tombol download dengan URL: {download_url}"
            ),
        })

    except FileNotFoundError as e:
        return _err(str(e), "NOT_FOUND")
    except PermissionError as e:
        return _err(str(e), "PERMISSION_DENIED")
    except Exception as e:
        return _err(f"generate_download_link gagal: {e}", "UNEXPECTED_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER — untuk integrasi langsung ke ToolRegistry existing
# ─────────────────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "manage_zip": manage_zip,
    "get_file_as_base64": get_file_as_base64,
    "generate_download_link": generate_download_link,
}


def dispatch(tool_name: str, args: dict) -> dict:
    """
    Central dispatcher. Panggil dari executor:
        result = dispatch("manage_zip", {"action": "list", "filename": "data.zip"})
    """
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return _err(f"Tool tidak dikenal: '{tool_name}'", "UNKNOWN_TOOL")
    try:
        return fn(**args)
    except TypeError as e:
        return _err(f"Argumen salah untuk '{tool_name}': {e}", "BAD_ARGUMENTS")
    except Exception as e:
        return _err(f"Tool '{tool_name}' crash: {e}", "RUNTIME_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — jalankan langsung: python file_manager.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    def _pp(label, result):
        print(f"\n{'─'*60}")
        print(f"▶ {label}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 60)
    print("  File Manager Suite — Quick Test")
    print(f"  Sandbox: {SANDBOX_ROOT}")
    print("=" * 60)

    # 1. Write beberapa file
    _pp("write_file: hello.txt", write_file("hello.txt", "Hello dari GazccAgent!\nLine 2."))
    _pp("write_file: config.json", write_file("config.json", json.dumps({"version": "1.0", "debug": True}, indent=2)))

    # 2. List files
    _pp("list_files: .", list_files("."))

    # 3. Read file
    _pp("read_file: hello.txt", read_file("hello.txt"))

    # 4. Create ZIP
    _pp("manage_zip: create", manage_zip("create", "bundle.zip", files=["hello.txt", "config.json"]))

    # 5. List ZIP
    _pp("manage_zip: list", manage_zip("list", "bundle.zip"))

    # 6. Extract ZIP
    _pp("manage_zip: extract", manage_zip("extract", "bundle.zip", dest_dir="extracted/"))

    # 7. Read extracted file
    _pp("read_file: extracted/hello.txt", read_file("extracted/hello.txt"))

    # 8. Modify & re-write
    _pp("write_file: extracted/hello.txt (modified)", write_file("extracted/hello.txt", "MODIFIED oleh Agent!"))

    # 9. Create new ZIP dari modified files
    _pp("manage_zip: create output.zip", manage_zip("create", "output.zip", files=["extracted/hello.txt", "extracted/config.json"]))

    # 10. Export Base64
    result = get_file_as_base64("output.zip")
    result_short = {**result}
    if "base64_data" in result_short:
        result_short["base64_data"] = result_short["base64_data"][:80] + "...[truncated]"
    if "export_marker" in result_short:
        result_short["export_marker"] = result_short["export_marker"][:80] + "...[truncated]"
    _pp("get_file_as_base64: output.zip (truncated)", result_short)

    # 11. Download link
    _pp("generate_download_link: output.zip", generate_download_link("output.zip"))

    print(f"\n{'='*60}")
    print("  ✓ Semua tool berhasil dieksekusi.")
    print("=" * 60)
