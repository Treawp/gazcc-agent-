"""
agent/extra_tools.py
GazccAgent — Extra Tools Pack
Tools: SQLite, DataViz, TokenCounter, Screenshot, Whois, DNS,
       URLMeta, MarkdownToHTML, HTMLToMarkdown, OCR,
       ImageResize, Git, Weather, CronScheduler
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import socket
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .tools import BaseTool, ToolResult
except ImportError:
    class ToolResult:
        def __init__(self, success: bool, output: str, metadata: dict | None = None):
            self.success = success
            self.output = output
            self.metadata = metadata or {}
    class BaseTool:
        name = ""
        description = ""
        parameters = ""
        async def run(self, *a, **kw) -> ToolResult:
            return ToolResult(False, "not implemented")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SQLiteTool — create/query SQLite database
# ══════════════════════════════════════════════════════════════════════════════

class SQLiteTool(BaseTool):
    name = "sqlite"
    description = (
        "Run SQL on a SQLite database file. "
        "Actions: query (SELECT), execute (INSERT/UPDATE/DELETE/CREATE), "
        "tables (list all tables), schema (show table schema)."
    )
    parameters = "action: str, db_path: str, sql: str = '', table: str = ''"

    async def run(self, action: str, db_path: str, sql: str = "", table: str = "") -> ToolResult:
        try:
            db_path = os.path.expandvars(db_path)
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if action == "tables":
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                rows = [r[0] for r in cur.fetchall()]
                conn.close()
                return ToolResult(True, "Tables: " + ", ".join(rows) if rows else "No tables found.",
                                  {"count": len(rows)})

            elif action == "schema":
                if not table:
                    return ToolResult(False, "table parameter required for schema action")
                cur.execute(f"PRAGMA table_info({table})")
                cols = cur.fetchall()
                conn.close()
                if not cols:
                    return ToolResult(False, f"Table '{table}' not found")
                lines = [f"  {c['name']} {c['type']} {'NOT NULL' if c['notnull'] else ''} {'PK' if c['pk'] else ''}".strip()
                         for c in cols]
                return ToolResult(True, f"Schema for {table}:\n" + "\n".join(lines))

            elif action == "query":
                if not sql:
                    return ToolResult(False, "sql parameter required")
                cur.execute(sql)
                rows = cur.fetchall()
                conn.close()
                if not rows:
                    return ToolResult(True, "Query returned 0 rows.", {"rows": 0})
                keys = rows[0].keys()
                lines = ["\t".join(keys)]
                for r in rows[:200]:
                    lines.append("\t".join(str(r[k]) for k in keys))
                return ToolResult(True, "\n".join(lines), {"rows": len(rows)})

            elif action == "execute":
                if not sql:
                    return ToolResult(False, "sql parameter required")
                cur.execute(sql)
                conn.commit()
                affected = cur.rowcount
                conn.close()
                return ToolResult(True, f"OK. Rows affected: {affected}", {"rows_affected": affected})

            else:
                conn.close()
                return ToolResult(False, f"Unknown action '{action}'. Use: query, execute, tables, schema")

        except Exception as e:
            return ToolResult(False, f"SQLite error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DataVizTool — generate chart as base64 PNG
# ══════════════════════════════════════════════════════════════════════════════

class DataVizTool(BaseTool):
    name = "data_viz"
    description = (
        "Generate a chart (bar, line, pie, scatter) and save as PNG. "
        "Pass labels and values as JSON arrays. Returns file path + base64 preview."
    )
    parameters = (
        "chart_type: str, labels: list, values: list, "
        "title: str = '', output_path: str = '/tmp/chart.png', "
        "series_name: str = 'Data'"
    )

    async def run(
        self,
        chart_type: str,
        labels: list,
        values: list,
        title: str = "",
        output_path: str = "/tmp/chart.png",
        series_name: str = "Data",
    ) -> ToolResult:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")

            colors = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#ffa657",
                      "#79c0ff", "#56d364", "#f78166", "#b392f0", "#e3b341"]

            ct = chart_type.lower()
            if ct == "bar":
                ax.bar(labels, values, color=colors[:len(labels)])
                ax.set_xlabel("", color="#8b949e")
                ax.set_ylabel(series_name, color="#8b949e")
            elif ct == "line":
                ax.plot(labels, values, color="#58a6ff", linewidth=2, marker="o",
                        markersize=5, markerfacecolor="#ff7b72")
                ax.set_ylabel(series_name, color="#8b949e")
                ax.fill_between(range(len(labels)), values, alpha=0.1, color="#58a6ff")
            elif ct == "pie":
                ax.pie(values, labels=labels, colors=colors[:len(labels)],
                       autopct="%1.1f%%", textprops={"color": "#c9d1d9"})
            elif ct == "scatter":
                ax.scatter(labels, values, color="#58a6ff", s=60, alpha=0.8)
                ax.set_ylabel(series_name, color="#8b949e")
            else:
                return ToolResult(False, f"Unknown chart_type '{chart_type}'. Use: bar, line, pie, scatter")

            if title:
                ax.set_title(title, color="#c9d1d9", fontsize=13, pad=12)
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            plt.tight_layout()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)

            with open(output_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            return ToolResult(True, f"Chart saved: {output_path}\n[IMG_BASE64:{b64[:80]}...]",
                              {"path": output_path, "base64_preview": b64[:200]})
        except ImportError:
            return ToolResult(False, "matplotlib not installed. Run: pip install matplotlib")
        except Exception as e:
            return ToolResult(False, f"DataViz error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TokenCounterTool — count tokens (tiktoken or char estimate)
# ══════════════════════════════════════════════════════════════════════════════

class TokenCounterTool(BaseTool):
    name = "token_counter"
    description = (
        "Count tokens in a text string. Uses tiktoken if available, "
        "otherwise estimates (~4 chars per token). "
        "Useful to check if content fits in context window before sending to LLM."
    )
    parameters = "text: str, model: str = 'gpt-4o'"

    async def run(self, text: str, model: str = "gpt-4o") -> ToolResult:
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            count = len(tokens)
            method = "tiktoken"
        except ImportError:
            count = max(1, len(text) // 4)
            method = "estimate (~4 chars/token)"

        limits = {
            "gpt-4o": 128000, "gpt-4o-mini": 128000,
            "claude-3-5-sonnet": 200000, "claude-3-haiku": 200000,
            "qwen3.6-plus": 1000000, "deepseek-v3": 64000,
            "gemini-2.0-flash": 1048576,
        }
        ctx = next((v for k, v in limits.items() if k in model.lower()), 32000)
        pct = round(count / ctx * 100, 1)

        return ToolResult(True,
            f"Tokens: {count:,} / {ctx:,} ({pct}%) — method: {method}",
            {"tokens": count, "context_limit": ctx, "percent_used": pct})


# ══════════════════════════════════════════════════════════════════════════════
# 6. ScreenshotTool — screenshot webpage to PNG
# ══════════════════════════════════════════════════════════════════════════════

class ScreenshotTool(BaseTool):
    name = "screenshot"
    description = (
        "Take a screenshot of a webpage URL and save as PNG. "
        "Requires playwright (pip install playwright && playwright install chromium). "
        "Falls back to a simple HTML snapshot if playwright unavailable."
    )
    parameters = "url: str, output_path: str = '/tmp/screenshot.png', width: int = 1280, height: int = 800"

    async def run(
        self,
        url: str,
        output_path: str = "/tmp/screenshot.png",
        width: int = 1280,
        height: int = 800,
    ) -> ToolResult:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": width, "height": height})
                await page.goto(url, timeout=30000, wait_until="networkidle")
                await page.screenshot(path=output_path, full_page=False)
                await browser.close()
            size = os.path.getsize(output_path)
            return ToolResult(True,
                f"Screenshot saved: {output_path} ({size//1024}KB)",
                {"path": output_path, "url": url, "size_bytes": size})
        except ImportError:
            # Fallback: use httpx to fetch HTML snapshot
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, follow_redirects=True)
                    html_path = output_path.replace(".png", ".html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(r.text)
                return ToolResult(True,
                    f"Playwright not installed — saved HTML snapshot instead: {html_path}",
                    {"path": html_path, "url": url, "note": "install playwright for PNG screenshots"})
            except Exception as e:
                return ToolResult(False, f"Screenshot failed: {e}")
        except Exception as e:
            return ToolResult(False, f"Screenshot error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. WhoisTool — WHOIS domain lookup
# ══════════════════════════════════════════════════════════════════════════════

class WhoisTool(BaseTool):
    name = "whois"
    description = (
        "WHOIS lookup for a domain. Returns registrar, creation date, "
        "expiry date, name servers, and registrant info."
    )
    parameters = "domain: str"

    async def run(self, domain: str) -> ToolResult:
        domain = domain.strip().lower().lstrip("https://").lstrip("http://").split("/")[0]
        try:
            import whois as pywhois
            w = pywhois.whois(domain)
            lines = []
            for key in ["domain_name", "registrar", "creation_date", "expiration_date",
                        "updated_date", "name_servers", "status", "emails"]:
                val = getattr(w, key, None)
                if val:
                    if isinstance(val, list):
                        val = val[0] if len(val) == 1 else str(val[:3])
                    lines.append(f"{key}: {val}")
            return ToolResult(True, "\n".join(lines) if lines else "No WHOIS data found",
                              {"domain": domain})
        except ImportError:
            # Fallback: raw socket WHOIS
            return await self._raw_whois(domain)
        except Exception as e:
            return await self._raw_whois(domain)

    async def _raw_whois(self, domain: str) -> ToolResult:
        try:
            tld = domain.rsplit(".", 1)[-1]
            server_map = {
                "com": "whois.verisign-grs.com", "net": "whois.verisign-grs.com",
                "org": "whois.publicinterestregistry.org", "io": "whois.nic.io",
                "id": "whois.pandi.or.id", "co": "whois.nic.co",
            }
            server = server_map.get(tld, f"whois.nic.{tld}")
            loop = asyncio.get_event_loop()
            def _query():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(10)
                    s.connect((server, 43))
                    s.send((domain + "\r\n").encode())
                    data = b""
                    while True:
                        chunk = s.recv(4096)
                        if not chunk: break
                        data += chunk
                    s.close()
                    return data.decode("utf-8", errors="ignore")
                except Exception as e:
                    return f"Error: {e}"
            raw = await loop.run_in_executor(None, _query)
            if raw.startswith("Error"):
                return ToolResult(False, f"WHOIS raw query failed: {raw}\nHint: pip install python-whois")
            # Parse key fields
            result = []
            for line in raw.splitlines():
                line = line.strip()
                if any(k in line.lower() for k in ["registrar", "creation", "expir", "name server", "updated", "status"]):
                    result.append(line)
            return ToolResult(True, "\n".join(result[:20]) or raw[:800], {"domain": domain})
        except Exception as e:
            return ToolResult(False, f"WHOIS failed: {e}. Hint: pip install python-whois")


# ══════════════════════════════════════════════════════════════════════════════
# 9. DNSLookupTool — DNS record query
# ══════════════════════════════════════════════════════════════════════════════

class DNSLookupTool(BaseTool):
    name = "dns_lookup"
    description = (
        "DNS lookup for a domain. Supports record types: "
        "A, AAAA, MX, TXT, CNAME, NS, SOA. "
        "Uses dnspython if available, falls back to socket."
    )
    parameters = "domain: str, record_type: str = 'A'"

    async def run(self, domain: str, record_type: str = "A") -> ToolResult:
        domain = domain.strip().lower().lstrip("https://").lstrip("http://").split("/")[0]
        record_type = record_type.upper()
        try:
            import dns.resolver
            answers = dns.resolver.resolve(domain, record_type)
            results = [str(r) for r in answers]
            return ToolResult(True,
                f"{record_type} records for {domain}:\n" + "\n".join(results),
                {"domain": domain, "type": record_type, "count": len(results)})
        except ImportError:
            # Fallback for A records only
            if record_type == "A":
                try:
                    loop = asyncio.get_event_loop()
                    addrs = await loop.run_in_executor(None, socket.gethostbyname_ex, domain)
                    return ToolResult(True,
                        f"A records for {domain}: {', '.join(addrs[2])}",
                        {"domain": domain, "type": "A"})
                except Exception as e:
                    return ToolResult(False, f"DNS lookup failed: {e}")
            return ToolResult(False,
                f"dnspython not installed (only A fallback available). "
                f"Run: pip install dnspython\nFor A record, omit record_type.")
        except Exception as e:
            return ToolResult(False, f"DNS lookup error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. URLMetaTool — extract Open Graph / meta tags from URL
# ══════════════════════════════════════════════════════════════════════════════

class URLMetaTool(BaseTool):
    name = "url_meta"
    description = (
        "Extract metadata from a URL: title, description, Open Graph tags, "
        "keywords, author, canonical URL, favicons, and language. "
        "Faster than full fetch_url as it only reads the <head> section."
    )
    parameters = "url: str"

    async def run(self, url: str) -> ToolResult:
        try:
            import httpx
            headers = {"User-Agent": "Mozilla/5.0 (compatible; GazccBot/1.0)"}
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                html = r.text[:50000]  # only need head

            meta = {}
            # Title
            m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
            if m: meta["title"] = m.group(1).strip()

            # Standard meta tags
            for tag in re.finditer(r'<meta\s+([^>]+)>', html, re.I | re.S):
                attrs = tag.group(1)
                name_m = re.search(r'(?:name|property)\s*=\s*["\']([^"\']+)["\']', attrs, re.I)
                cont_m = re.search(r'content\s*=\s*["\']([^"\']*)["\']', attrs, re.I)
                if name_m and cont_m:
                    key = name_m.group(1).lower()
                    val = cont_m.group(1).strip()
                    if val and key in ["description", "keywords", "author",
                                       "og:title", "og:description", "og:image",
                                       "og:url", "og:type", "og:site_name",
                                       "twitter:title", "twitter:description",
                                       "twitter:image", "robots"]:
                        meta[key] = val

            # Canonical
            can = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', html, re.I)
            if can: meta["canonical"] = can.group(1)

            # Lang
            lang = re.search(r'<html[^>]+lang=["\']([^"\']+)["\']', html, re.I)
            if lang: meta["language"] = lang.group(1)

            if not meta:
                return ToolResult(False, "No metadata found on page")

            lines = [f"{k}: {v}" for k, v in meta.items()]
            return ToolResult(True, "\n".join(lines), {"url": url, "fields": len(meta)})

        except Exception as e:
            return ToolResult(False, f"URLMeta error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MarkdownToHTMLTool — convert Markdown to HTML
# ══════════════════════════════════════════════════════════════════════════════

class MarkdownToHTMLTool(BaseTool):
    name = "md_to_html"
    description = (
        "Convert Markdown text to HTML. "
        "Optionally wrap in a full HTML document with dark-mode styling. "
        "Can also save output to a file."
    )
    parameters = "markdown: str, full_page: bool = False, output_path: str = ''"

    async def run(self, markdown: str, full_page: bool = False, output_path: str = "") -> ToolResult:
        try:
            import markdown as md_lib
            html_body = md_lib.markdown(markdown, extensions=["tables", "fenced_code", "nl2br"])
        except ImportError:
            # Fallback: basic regex conversion
            html_body = markdown
            html_body = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_body, flags=re.M)
            html_body = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_body, flags=re.M)
            html_body = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_body, flags=re.M)
            html_body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_body)
            html_body = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_body)
            html_body = re.sub(r'`(.+?)`', r'<code>\1</code>', html_body)
            html_body = re.sub(r'\n\n', '</p><p>', html_body)
            html_body = f"<p>{html_body}</p>"

        if full_page:
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body{{background:#0d1117;color:#c9d1d9;font-family:system-ui,sans-serif;
       max-width:800px;margin:40px auto;padding:0 20px;line-height:1.7}}
  h1,h2,h3{{color:#e6edf3}} a{{color:#58a6ff}}
  code{{background:#161b22;padding:2px 6px;border-radius:4px;font-size:.9em}}
  pre{{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}}
  table{{border-collapse:collapse;width:100%}}
  td,th{{border:1px solid #30363d;padding:8px 12px}}
  th{{background:#161b22}}
  blockquote{{border-left:4px solid #30363d;padding-left:16px;color:#8b949e;margin:0}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
        else:
            html = html_body

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            return ToolResult(True, f"HTML saved: {output_path} ({len(html)} chars)",
                              {"path": output_path, "chars": len(html)})

        preview = html[:1000] + ("..." if len(html) > 1000 else "")
        return ToolResult(True, preview, {"chars": len(html)})


# ══════════════════════════════════════════════════════════════════════════════
# 12. HTMLToMarkdownTool — convert HTML to clean Markdown
# ══════════════════════════════════════════════════════════════════════════════

class HTMLToMarkdownTool(BaseTool):
    name = "html_to_md"
    description = (
        "Convert HTML string or fetched URL to clean Markdown text. "
        "Strips scripts, styles, nav. Great for cleaning scraped content."
    )
    parameters = "html: str = '', url: str = '', output_path: str = ''"

    async def run(self, html: str = "", url: str = "", output_path: str = "") -> ToolResult:
        try:
            if url and not html:
                import httpx
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    r = await client.get(url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; GazccBot/1.0)"})
                    html = r.text

            if not html:
                return ToolResult(False, "Provide html or url parameter")

            try:
                import html2text
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                md = h.handle(html)
            except ImportError:
                # Manual fallback
                md = html
                md = re.sub(r'<script[^>]*>.*?</script>', '', md, flags=re.S | re.I)
                md = re.sub(r'<style[^>]*>.*?</style>', '', md, flags=re.S | re.I)
                md = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', lambda m: '#'*int(m.group(1))+' '+m.group(2)+'\n', md, flags=re.I|re.S)
                md = re.sub(r'<(b|strong)[^>]*>(.*?)</\1>', r'**\2**', md, flags=re.I|re.S)
                md = re.sub(r'<(i|em)[^>]*>(.*?)</\1>', r'*\2*', md, flags=re.I|re.S)
                md = re.sub(r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', md, flags=re.I|re.S)
                md = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', md, flags=re.I|re.S)
                md = re.sub(r'<br\s*/?>', '\n', md, flags=re.I)
                md = re.sub(r'<p[^>]*>', '\n\n', md, flags=re.I)
                md = re.sub(r'<[^>]+>', '', md)
                import html as html_mod
                md = html_mod.unescape(md)
                md = re.sub(r'\n{3,}', '\n\n', md).strip()

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(md)
                return ToolResult(True, f"Markdown saved: {output_path} ({len(md)} chars)",
                                  {"path": output_path, "chars": len(md)})

            preview = md[:2000] + ("..." if len(md) > 2000 else "")
            return ToolResult(True, preview, {"chars": len(md)})

        except Exception as e:
            return ToolResult(False, f"HTML to Markdown error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 14. OCRTool — extract text from image via pytesseract
# ══════════════════════════════════════════════════════════════════════════════

class OCRTool(BaseTool):
    name = "ocr"
    description = (
        "Extract text from an image file using OCR (pytesseract). "
        "Supports: PNG, JPG, WEBP, BMP, TIFF. "
        "Requires: pip install pytesseract pillow + tesseract-ocr system package."
    )
    parameters = "image_path: str, lang: str = 'eng'"

    async def run(self, image_path: str, lang: str = "eng") -> ToolResult:
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter

            img = Image.open(image_path)
            # Pre-process for better accuracy
            if img.mode != "L":
                img = img.convert("L")
            img = img.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(img, lang=lang, config="--psm 3")
            )
            text = text.strip()
            if not text:
                return ToolResult(False, "No text detected in image.")
            return ToolResult(True, text, {"chars": len(text), "lang": lang})

        except ImportError as e:
            return ToolResult(False,
                f"Missing dependency: {e}\n"
                "Run: pip install pytesseract pillow\n"
                "Also install tesseract: apt install tesseract-ocr (Linux) or brew install tesseract (Mac)")
        except FileNotFoundError:
            return ToolResult(False, f"Image file not found: {image_path}")
        except Exception as e:
            return ToolResult(False, f"OCR error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 15. ImageResizeTool — resize / compress images via Pillow
# ══════════════════════════════════════════════════════════════════════════════

class ImageResizeTool(BaseTool):
    name = "image_resize"
    description = (
        "Resize, crop, rotate, or convert an image. "
        "Actions: resize (set width/height), thumbnail (max dimension), "
        "crop (x,y,w,h), rotate (degrees), convert (format: PNG/JPEG/WEBP), "
        "info (get image metadata)."
    )
    parameters = (
        "image_path: str, action: str, output_path: str = '', "
        "width: int = 0, height: int = 0, max_dim: int = 512, "
        "x: int = 0, y: int = 0, degrees: float = 0, "
        "format: str = 'JPEG', quality: int = 85"
    )

    async def run(
        self,
        image_path: str,
        action: str,
        output_path: str = "",
        width: int = 0,
        height: int = 0,
        max_dim: int = 512,
        x: int = 0,
        y: int = 0,
        degrees: float = 0,
        format: str = "JPEG",
        quality: int = 85,
    ) -> ToolResult:
        try:
            from PIL import Image as PILImage, ExifTags

            img = PILImage.open(image_path)
            orig_size = img.size
            orig_mode = img.mode
            action = action.lower()

            if action == "info":
                info = {
                    "size": f"{img.width}x{img.height}",
                    "mode": img.mode,
                    "format": img.format,
                    "file_size": f"{os.path.getsize(image_path)//1024}KB",
                }
                return ToolResult(True, "\n".join(f"{k}: {v}" for k, v in info.items()), info)

            elif action == "resize":
                if not width and not height:
                    return ToolResult(False, "Provide width and/or height")
                if not width:
                    ratio = height / img.height
                    width = int(img.width * ratio)
                if not height:
                    ratio = width / img.width
                    height = int(img.height * ratio)
                img = img.resize((width, height), PILImage.LANCZOS)

            elif action == "thumbnail":
                img.thumbnail((max_dim, max_dim), PILImage.LANCZOS)

            elif action == "crop":
                img = img.crop((x, y, x + width, y + height))

            elif action == "rotate":
                img = img.rotate(degrees, expand=True)

            elif action == "convert":
                if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

            else:
                return ToolResult(False, f"Unknown action '{action}'. Use: resize, thumbnail, crop, rotate, convert, info")

            if not output_path:
                ext = format.lower() if action == "convert" else Path(image_path).suffix
                output_path = str(Path(image_path).with_suffix(f".out{ext}"))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {}
            if format.upper() in ("JPEG", "WEBP"):
                save_kwargs["quality"] = quality
            img.save(output_path, format=format.upper(), **save_kwargs)
            new_size = os.path.getsize(output_path)

            return ToolResult(True,
                f"Done. {orig_size[0]}x{orig_size[1]} → {img.width}x{img.height} | "
                f"Saved: {output_path} ({new_size//1024}KB)",
                {"path": output_path, "width": img.width, "height": img.height})

        except ImportError:
            return ToolResult(False, "Pillow not installed. Run: pip install pillow")
        except Exception as e:
            return ToolResult(False, f"ImageResize error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 16. GitTool — git operations in working directory
# ══════════════════════════════════════════════════════════════════════════════

class GitTool(BaseTool):
    name = "git"
    description = (
        "Run git operations in a repository directory. "
        "Actions: init, status, add, commit, log, diff, branch, "
        "checkout, clone, pull, push, stash."
    )
    parameters = "action: str, path: str = '.', args: str = ''"

    async def run(self, action: str, path: str = ".", args: str = "") -> ToolResult:
        action = action.lower().strip()
        safe_actions = {
            "init":     ["git", "init"],
            "status":   ["git", "status", "--short"],
            "log":      ["git", "log", "--oneline", "-20"],
            "diff":     ["git", "diff"],
            "branch":   ["git", "branch", "-a"],
            "stash":    ["git", "stash", "list"],
        }
        arg_actions = {
            "add":      lambda a: ["git", "add"] + (a.split() if a else ["."]),
            "commit":   lambda a: ["git", "commit", "-m", a or "auto commit by GazccAgent"],
            "checkout": lambda a: ["git", "checkout"] + a.split(),
            "clone":    lambda a: ["git", "clone"] + a.split(),
            "pull":     lambda a: ["git", "pull"] + (a.split() if a else []),
            "push":     lambda a: ["git", "push"] + (a.split() if a else []),
            "stash_push": lambda a: ["git", "stash", "push", "-m", a or "gazcc stash"],
            "stash_pop":  lambda a: ["git", "stash", "pop"],
        }

        if action in safe_actions:
            cmd = safe_actions[action]
        elif action in arg_actions:
            cmd = arg_actions[action](args)
        else:
            return ToolResult(False,
                f"Unknown git action '{action}'. "
                "Use: init, status, add, commit, log, diff, branch, checkout, clone, pull, push, stash")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                cmd,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=60,
            ))
            out = (result.stdout + result.stderr).strip()
            success = result.returncode == 0
            return ToolResult(success, out or "(no output)", {"returncode": result.returncode})
        except FileNotFoundError:
            return ToolResult(False, "git not found. Install git first.")
        except subprocess.TimeoutExpired:
            return ToolResult(False, "git command timed out (60s)")
        except Exception as e:
            return ToolResult(False, f"Git error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 18. WeatherTool — realtime weather via Open-Meteo (free, no API key)
# ══════════════════════════════════════════════════════════════════════════════

class WeatherTool(BaseTool):
    name = "weather"
    description = (
        "Get current weather and 3-day forecast for any city. "
        "Uses Open-Meteo (free, no API key required). "
        "Also supports latitude/longitude directly."
    )
    parameters = "city: str = '', lat: float = 0, lon: float = 0, forecast_days: int = 3"

    _WMO = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
        71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Showers",
        81: "Heavy showers", 82: "Violent showers", 95: "Thunderstorm",
        96: "Thunderstorm w/ hail", 99: "Severe thunderstorm",
    }

    async def run(self, city: str = "", lat: float = 0, lon: float = 0,
                  forecast_days: int = 3) -> ToolResult:
        try:
            import httpx

            if city and not (lat and lon):
                # Geocode city
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.get(
                        "https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1, "language": "en", "format": "json"}
                    )
                    geo = r.json()
                results = geo.get("results")
                if not results:
                    return ToolResult(False, f"City '{city}' not found in geocoding API")
                loc = results[0]
                lat, lon = loc["latitude"], loc["longitude"]
                location_name = f"{loc.get('name', city)}, {loc.get('country', '')}"
            else:
                location_name = f"{lat}, {lon}"

            if not lat and not lon:
                return ToolResult(False, "Provide city name or lat/lon coordinates")

            fd = max(1, min(forecast_days, 7))
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat, "longitude": lon,
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weathercode,apparent_temperature",
                        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                        "timezone": "auto",
                        "forecast_days": fd,
                    }
                )
                data = r.json()

            cur = data.get("current", {})
            daily = data.get("daily", {})
            cur_vals = cur.get("variables", {}) if "variables" in cur else cur

            lines = [f"📍 {location_name}"]
            lines.append(f"🌡  Current: {cur.get('temperature_2m', cur.get('temperature_2m', '?'))}°C "
                         f"(feels {cur.get('apparent_temperature', '?')}°C)")
            lines.append(f"💧 Humidity: {cur.get('relative_humidity_2m', '?')}%")
            lines.append(f"💨 Wind: {cur.get('wind_speed_10m', '?')} km/h")
            wc = cur.get("weathercode", 0)
            lines.append(f"☁  Condition: {self._WMO.get(wc, f'Code {wc}')}")

            lines.append("\n📅 Forecast:")
            dates = daily.get("time", [])
            t_max = daily.get("temperature_2m_max", [])
            t_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            wcs = daily.get("weathercode", [])
            for i, d in enumerate(dates):
                cond = self._WMO.get(wcs[i] if i < len(wcs) else 0, "?")
                lines.append(
                    f"  {d}: {t_min[i] if i<len(t_min) else '?'}°C – "
                    f"{t_max[i] if i<len(t_max) else '?'}°C | "
                    f"{precip[i] if i<len(precip) else 0}mm | {cond}"
                )

            return ToolResult(True, "\n".join(lines),
                              {"lat": lat, "lon": lon, "city": location_name})

        except Exception as e:
            return ToolResult(False, f"Weather error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 20. CronSchedulerTool — schedule recurring tasks (file-backed)
# ══════════════════════════════════════════════════════════════════════════════

class CronSchedulerTool(BaseTool):
    name = "cron_scheduler"
    description = (
        "Schedule, list, and manage recurring tasks. "
        "Actions: add (create schedule), list (all tasks), "
        "remove (delete by id), due (check which tasks are due now), "
        "clear (delete all). Tasks persist to file between runs."
    )
    parameters = (
        "action: str, "
        "task: str = '', "
        "interval_minutes: int = 60, "
        "task_id: str = '', "
        "storage_path: str = '/tmp/gazcc_cron.json'"
    )

    def _load(self, path: str) -> list:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, path: str, tasks: list):
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2)

    async def run(
        self,
        action: str,
        task: str = "",
        interval_minutes: int = 60,
        task_id: str = "",
        storage_path: str = "/tmp/gazcc_cron.json",
    ) -> ToolResult:
        try:
            action = action.lower()
            tasks = self._load(storage_path)
            now = time.time()

            if action == "add":
                if not task:
                    return ToolResult(False, "task parameter required")
                tid = f"task_{int(now)}_{len(tasks)}"
                entry = {
                    "id": tid,
                    "task": task,
                    "interval_minutes": interval_minutes,
                    "created_at": datetime.utcnow().isoformat(),
                    "next_run": now + interval_minutes * 60,
                    "last_run": None,
                    "run_count": 0,
                }
                tasks.append(entry)
                self._save(storage_path, tasks)
                return ToolResult(True,
                    f"Scheduled: [{tid}] '{task}' every {interval_minutes}min\n"
                    f"Next run: {datetime.fromtimestamp(entry['next_run']).strftime('%Y-%m-%d %H:%M:%S')}",
                    {"id": tid})

            elif action == "list":
                if not tasks:
                    return ToolResult(True, "No scheduled tasks.", {"count": 0})
                lines = []
                for t in tasks:
                    nr = datetime.fromtimestamp(t["next_run"]).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"[{t['id']}] '{t['task']}' every {t['interval_minutes']}min | next: {nr} | runs: {t['run_count']}")
                return ToolResult(True, "\n".join(lines), {"count": len(tasks)})

            elif action == "remove":
                if not task_id:
                    return ToolResult(False, "task_id parameter required")
                before = len(tasks)
                tasks = [t for t in tasks if t["id"] != task_id]
                if len(tasks) == before:
                    return ToolResult(False, f"Task '{task_id}' not found")
                self._save(storage_path, tasks)
                return ToolResult(True, f"Removed task '{task_id}'", {"remaining": len(tasks)})

            elif action == "due":
                due = [t for t in tasks if t["next_run"] <= now]
                if not due:
                    return ToolResult(True, "No tasks due right now.", {"due_count": 0})
                # Update next_run for due tasks
                for t in tasks:
                    if t["next_run"] <= now:
                        t["last_run"] = now
                        t["run_count"] += 1
                        t["next_run"] = now + t["interval_minutes"] * 60
                self._save(storage_path, tasks)
                lines = [f"[{t['id']}] {t['task']}" for t in due]
                return ToolResult(True,
                    f"{len(due)} task(s) due:\n" + "\n".join(lines),
                    {"due_count": len(due), "tasks": [t["task"] for t in due]})

            elif action == "clear":
                self._save(storage_path, [])
                return ToolResult(True, f"Cleared {len(tasks)} scheduled task(s).", {"cleared": len(tasks)})

            else:
                return ToolResult(False,
                    f"Unknown action '{action}'. Use: add, list, remove, due, clear")

        except Exception as e:
            return ToolResult(False, f"CronScheduler error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER — call this from core.py or wherever ToolRegistry is initialized
# ══════════════════════════════════════════════════════════════════════════════

EXTRA_TOOLS = [
    SQLiteTool,
    DataVizTool,
    TokenCounterTool,
    ScreenshotTool,
    WhoisTool,
    DNSLookupTool,
    URLMetaTool,
    MarkdownToHTMLTool,
    HTMLToMarkdownTool,
    OCRTool,
    ImageResizeTool,
    GitTool,
    WeatherTool,
    CronSchedulerTool,
]


def register_extra_tools(registry, cfg: dict | None = None) -> None:
    """
    Register all extra tools into an existing ToolRegistry instance.

    USAGE in agent/core.py:
        from .extra_tools import register_extra_tools
        register_extra_tools(self._tools, self._cfg)
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})
    if not tool_cfg.get("extra_tools", True):
        return
    for cls in EXTRA_TOOLS:
        tool = cls()
        registry._register(tool)
