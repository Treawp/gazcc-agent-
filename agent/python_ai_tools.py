"""
agent/python_ai_tools.py
GazccAgent — 32 Python AI Tools Pack
Follows exact pattern of extra_tools.py.

USAGE in agent/core.py:
    from .python_ai_tools import register_python_ai_tools
    register_python_ai_tools(self._tools, self._cfg)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
import secrets
import socket
import ssl
import string
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

try:
    from .tools import BaseTool, ToolResult
except ImportError:
    class ToolResult:
        def __init__(self, success: bool, output: str, metadata: dict | None = None):
            self.success = success; self.output = output; self.metadata = metadata or {}
        def __str__(self): return f"{'✓' if self.success else '✗'} {self.output}"
    class BaseTool:
        name = ""; description = ""; parameters = ""
        async def run(self, *a, **kw) -> ToolResult: return ToolResult(False, "not implemented")


def _try_import(mod: str):
    try:
        import importlib
        return importlib.import_module(mod)
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE TOOLS (1–5)
# ══════════════════════════════════════════════════════════════════════════════

class ExcelParseTool(BaseTool):
    name = "excel_parse"
    description = "Parse an Excel (.xlsx/.xls) file into JSON. Returns rows, sheet names, column stats, and errors."
    parameters = "file_path: str, sheet_name: str = None, clean_data: bool = True"

    async def run(self, file_path: str, sheet_name: str = None, clean_data: bool = True) -> ToolResult:
        pd = _try_import("pandas")
        if pd is None: return ToolResult(False, "Missing dep: pip install pandas openpyxl")
        try:
            p = Path(file_path)
            if not p.exists(): return ToolResult(False, f"File not found: {file_path}")
            xls = pd.ExcelFile(file_path); sheets = xls.sheet_names
            target = sheet_name or sheets[0]
            df = pd.read_excel(file_path, sheet_name=target)
            if clean_data:
                df = df.dropna(how="all").drop_duplicates()
                df.columns = [str(c).strip() for c in df.columns]
            stats = {"rows": len(df), "columns": len(df.columns), "nulls": int(df.isnull().sum().sum()), "dtypes": {c: str(df[c].dtype) for c in df.columns}}
            return ToolResult(True, json.dumps({"data": df.head(100).to_dict(orient="records"), "sheets": sheets, "active_sheet": target, "stats": stats, "errors": []}, default=str))
        except Exception as e: return ToolResult(False, f"excel_parse error: {e}")


class ApiTestTool(BaseTool):
    name = "api_test"
    description = "Test an HTTP API endpoint. Returns status_code, headers, body, response_time_ms, success."
    parameters = "url: str, method: str = 'GET', headers: dict = None, body: dict = None, timeout: int = 30"

    async def run(self, url: str, method: str = "GET", headers: dict = None, body: dict = None, timeout: int = 30) -> ToolResult:
        try:
            t0 = time.time()
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                resp = await c.request(method.upper(), url, headers=headers or {}, json=body)
            elapsed = round((time.time() - t0) * 1000, 2)
            try: rb = resp.json()
            except: rb = resp.text[:2000]
            r = {"status_code": resp.status_code, "headers": dict(resp.headers), "body": rb, "response_time_ms": elapsed, "success": 200 <= resp.status_code < 300, "url": url, "method": method.upper()}
            return ToolResult(r["success"], json.dumps(r, default=str))
        except httpx.TimeoutException: return ToolResult(False, f"api_test timeout after {timeout}s")
        except Exception as e: return ToolResult(False, f"api_test error: {e}")


class PdfMergeTool(BaseTool):
    name = "pdf_merge"
    description = "Merge multiple PDF files into one. Returns merged file path and page count."
    parameters = "file_paths: list, output_path: str = 'merged.pdf'"

    async def run(self, file_paths: list, output_path: str = "merged.pdf") -> ToolResult:
        pypdf = _try_import("pypdf")
        if pypdf is None: return ToolResult(False, "Missing dep: pip install pypdf")
        try:
            writer = pypdf.PdfWriter(); errors = []
            for fp in file_paths:
                p = Path(fp)
                if not p.exists(): errors.append(f"Not found: {fp}"); continue
                for page in pypdf.PdfReader(str(p)).pages: writer.add_page(page)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f: writer.write(f)
            return ToolResult(True, json.dumps({"output": output_path, "total_pages": len(writer.pages), "merged_files": len(file_paths)-len(errors), "errors": errors}))
        except Exception as e: return ToolResult(False, f"pdf_merge error: {e}")


class QrCodeGenerateTool(BaseTool):
    name = "qr_code_generate"
    description = "Generate a QR code PNG from text or URL. Returns output path and base64 data URI."
    parameters = "text: str, size: int = 300, color: str = 'black', bg: str = 'white', output_path: str = 'qrcode.png'"

    async def run(self, text: str, size: int = 300, color: str = "black", bg: str = "white", output_path: str = "qrcode.png") -> ToolResult:
        qrcode = _try_import("qrcode")
        if qrcode is None: return ToolResult(False, "Missing dep: pip install qrcode[pil]")
        try:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
            qr.add_data(text); qr.make(fit=True)
            img = qr.make_image(fill_color=color, back_color=bg)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return ToolResult(True, json.dumps({"output_path": output_path, "text": text, "base64_png": f"data:image/png;base64,{b64}"}))
        except Exception as e: return ToolResult(False, f"qr_code_generate error: {e}")


class PromptOptimizeTool(BaseTool):
    name = "prompt_optimize"
    description = "Optimize a prompt using heuristics. Goals: better_clarity, more_detailed, more_concise, technical, creative, step_by_step."
    parameters = "base_prompt: str, goal: str = 'better_clarity', context: dict = None"

    async def run(self, base_prompt: str, goal: str = "better_clarity", context: dict = None) -> ToolResult:
        if not base_prompt.strip(): return ToolResult(False, "base_prompt empty")
        valid = ["better_clarity","more_detailed","more_concise","technical","creative","step_by_step"]
        if goal not in valid: goal = "better_clarity"
        opt = base_prompt.strip(); suggestions = []
        if len(opt) < 20: suggestions.append("Prompt too short — add context")
        if goal == "more_detailed": opt += "\n\nInclude: detailed explanation, examples, edge cases. Return JSON format."; suggestions.append("Added detail requirements")
        elif goal == "step_by_step": opt = "Step by step:\n" + opt; suggestions.append("Added step-by-step framing")
        elif goal == "technical": opt += "\n\nInclude: input/output spec, error handling, constraints, code example."; suggestions.append("Added technical spec")
        elif goal == "more_concise": opt = ". ".join(opt.split(".")[:2]).strip() + "."; suggestions.append("Trimmed to core intent")
        elif goal == "creative": opt = "Write with vivid detail and creative flair: " + opt; suggestions.append("Added creative framing")
        else: opt = opt.rstrip(".") + ". Be specific about input, output format, and constraints."; suggestions.append("Added output format guidance")
        return ToolResult(True, json.dumps({"original": base_prompt, "optimized": opt, "goal": goal, "suggestions": suggestions, "char_delta": len(opt)-len(base_prompt)}))


# ══════════════════════════════════════════════════════════════════════════════
# DATA & ANALYTICS (6–10)
# ══════════════════════════════════════════════════════════════════════════════

class JsonSchemaGenerateTool(BaseTool):
    name = "json_schema_generate"
    description = "Auto-generate JSON Schema from a sample JSON string. Infers types for nested objects/arrays."
    parameters = "data_sample: str, output_path: str = None"

    def _infer(self, v):
        if v is None: return {"type": "null"}
        if isinstance(v, bool): return {"type": "boolean"}
        if isinstance(v, int): return {"type": "integer"}
        if isinstance(v, float): return {"type": "number"}
        if isinstance(v, str): return {"type": "string"}
        if isinstance(v, list):
            if not v: return {"type": "array", "items": {}}
            s = [self._infer(x) for x in v[:5]]; types = {x.get("type") for x in s}
            return {"type": "array", "items": s[0] if len(types)==1 else {"oneOf": s}}
        if isinstance(v, dict): return {"type": "object", "properties": {k: self._infer(vv) for k,vv in v.items()}, "required": list(v.keys())}
        return {"type": "string"}

    async def run(self, data_sample: str, output_path: str = None) -> ToolResult:
        try:
            data = json.loads(data_sample) if isinstance(data_sample, str) else data_sample
            schema = {"$schema": "http://json-schema.org/draft-07/schema#", **self._infer(data)}
            out = json.dumps(schema, indent=2)
            if output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(out)
            return ToolResult(True, out)
        except json.JSONDecodeError as e: return ToolResult(False, f"Invalid JSON: {e}")
        except Exception as e: return ToolResult(False, f"json_schema_generate error: {e}")


class DataCleanTool(BaseTool):
    name = "data_clean"
    description = "Clean JSON list/dict: remove nulls, deduplicate, strip special chars. Returns cleaned data + diff report."
    parameters = "data: str, remove_nulls: bool = True, remove_duplicates: bool = True, remove_special: bool = True"

    async def run(self, data: str, remove_nulls: bool = True, remove_duplicates: bool = True, remove_special: bool = True) -> ToolResult:
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            report = {"original_count": len(parsed) if isinstance(parsed, list) else 1, "operations": []}
            cv = lambda v: re.sub(r"[^\w\s\-.,@:/]", "", v).strip() if isinstance(v, str) and remove_special else v
            cr = lambda r: {k: cv(v) for k,v in {k: v for k,v in r.items() if v is not None and v != ""}.items()} if isinstance(r, dict) and remove_nulls else ({k: cv(v) for k,v in r.items()} if isinstance(r, dict) else r)
            if isinstance(parsed, list):
                cleaned = [cr(r) for r in parsed]
                if remove_nulls: prev=len(cleaned); cleaned=[r for r in cleaned if r]; report["operations"].append(f"removed_nulls: {prev-len(cleaned)}")
                if remove_duplicates:
                    seen, deduped = set(), []
                    for r in cleaned:
                        k = json.dumps(r, sort_keys=True)
                        if k not in seen: seen.add(k); deduped.append(r)
                    report["operations"].append(f"removed_duplicates: {len(cleaned)-len(deduped)}"); cleaned = deduped
                report["final_count"] = len(cleaned)
            else: cleaned = cr(parsed); report["final_count"] = 1
            return ToolResult(True, json.dumps({"data": cleaned, "report": report}))
        except Exception as e: return ToolResult(False, f"data_clean error: {e}")


class DataVizChartTool(BaseTool):
    name = "data_viz_chart"
    description = "Generate a chart PNG (bar/line/pie/scatter) from labels and values. Saves to output_path."
    parameters = "data_type: str, labels: list, values: list, title: str = 'Chart', output_path: str = 'chart.png'"

    async def run(self, data_type: str, labels: list, values: list, title: str = "Chart", output_path: str = "chart.png") -> ToolResult:
        mpl = _try_import("matplotlib.pyplot")
        if mpl is None: return ToolResult(False, "Missing dep: pip install matplotlib")
        try:
            if data_type not in ["bar","line","pie","scatter"]: return ToolResult(False, "data_type: bar|line|pie|scatter")
            if len(labels) != len(values): return ToolResult(False, f"labels({len(labels)}) != values({len(values)})")
            mpl.figure(figsize=(10,6))
            if data_type=="bar": mpl.bar(labels, values, color="#4F86C6"); mpl.xticks(rotation=45,ha="right")
            elif data_type=="line": mpl.plot(labels, values, marker="o", color="#4F86C6", linewidth=2); mpl.xticks(rotation=45,ha="right")
            elif data_type=="pie": mpl.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            elif data_type=="scatter": mpl.scatter(range(len(values)), values, color="#4F86C6", s=80); mpl.xticks(range(len(labels)), labels, rotation=45, ha="right")
            mpl.title(title, fontsize=14, fontweight="bold"); mpl.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            mpl.savefig(output_path, dpi=150, bbox_inches="tight"); mpl.close()
            return ToolResult(True, json.dumps({"output_path": output_path, "chart_type": data_type, "data_points": len(values)}))
        except Exception as e: return ToolResult(False, f"data_viz_chart error: {e}")


class CsvTransformTool(BaseTool):
    name = "csv_transform"
    description = "Convert CSV to json/excel/sql/csv. Returns output file path."
    parameters = "input_path: str, output_format: str = 'json', output_path: str = None"

    async def run(self, input_path: str, output_format: str = "json", output_path: str = None) -> ToolResult:
        pd = _try_import("pandas")
        if pd is None: return ToolResult(False, "Missing dep: pip install pandas openpyxl")
        try:
            p = Path(input_path)
            if not p.exists(): return ToolResult(False, f"Not found: {input_path}")
            df = pd.read_csv(input_path); fmt = output_format.lower()
            ext = {"json":".json","excel":".xlsx","sql":".sql","csv":".csv"}
            if fmt not in ext: return ToolResult(False, f"format: {list(ext.keys())}")
            if not output_path: output_path = str(p.with_suffix(ext[fmt]))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if fmt=="json": df.to_json(output_path, orient="records", indent=2)
            elif fmt=="excel": df.to_excel(output_path, index=False)
            elif fmt=="csv": df.to_csv(output_path, index=False)
            elif fmt=="sql":
                tbl = p.stem.replace(" ","_").lower()
                rows = [f"INSERT INTO `{tbl}` VALUES ({', '.join(repr(v) if isinstance(v,str) else str(v) for v in r)});" for _,r in df.iterrows()]
                Path(output_path).write_text("-- GazccAI SQL Export\n" + "\n".join(rows))
            return ToolResult(True, json.dumps({"input": input_path, "output": output_path, "format": fmt, "rows": len(df)}))
        except Exception as e: return ToolResult(False, f"csv_transform error: {e}")


class SqlGeneratorTool(BaseTool):
    name = "sql_generator"
    description = "Generate SQL from natural language + schema dict. Returns SQL string and saves to file if path given."
    parameters = "natural_query: str, db_schema: str, output_path: str = None"

    async def run(self, natural_query: str, db_schema: str, output_path: str = None) -> ToolResult:
        try:
            schema = json.loads(db_schema) if isinstance(db_schema, str) else db_schema
            q = natural_query.lower()
            tbl = next((t for t in (schema if isinstance(schema,dict) else {}) if t.lower() in q), None)
            if not tbl and isinstance(schema, dict): tbl = next(iter(schema), "table")
            cols = schema.get(tbl, ["*"]) if isinstance(schema, dict) else ["*"]
            col_str = ", ".join(cols) if cols != ["*"] else "*"
            if any(w in q for w in ["count","how many"]): sql = f"SELECT COUNT(*) AS total FROM {tbl};"
            elif any(w in q for w in ["insert","add new"]): sql = f"INSERT INTO {tbl} ({col_str}) VALUES ({', '.join(':'+c for c in cols)});"
            elif any(w in q for w in ["update","change","set"]): sql = f"UPDATE {tbl} SET {', '.join(c+' = :'+c for c in cols[:3])} WHERE id = :id;"
            elif any(w in q for w in ["delete","remove"]): sql = f"DELETE FROM {tbl} WHERE id = :id;"
            elif any(w in q for w in ["group","sum","avg"]): sql = f"SELECT {cols[0] if cols!=['*'] else 'col'}, COUNT(*) as cnt FROM {tbl} GROUP BY 1 ORDER BY cnt DESC;"
            else: sql = f"SELECT {col_str} FROM {tbl} ORDER BY id DESC LIMIT 100;"
            if output_path: Path(output_path).write_text(sql)
            return ToolResult(True, json.dumps({"sql": sql, "table": tbl, "natural_query": natural_query}))
        except Exception as e: return ToolResult(False, f"sql_generator error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY & AUTH (11–15)
# ══════════════════════════════════════════════════════════════════════════════

class PasswordGenerateTool(BaseTool):
    name = "password_generate"
    description = "Generate a cryptographically secure password. Returns password + strength info."
    parameters = "length: int = 16, use_special: bool = True, use_numbers: bool = True, use_upper: bool = True"

    async def run(self, length: int = 16, use_special: bool = True, use_numbers: bool = True, use_upper: bool = True) -> ToolResult:
        try:
            if not 4 <= length <= 256: return ToolResult(False, "length: 4–256")
            charset = string.ascii_lowercase; req = [secrets.choice(string.ascii_lowercase)]
            if use_upper: charset += string.ascii_uppercase; req.append(secrets.choice(string.ascii_uppercase))
            if use_numbers: charset += string.digits; req.append(secrets.choice(string.digits))
            if use_special: sp="!@#$%^&*()-_=+[]{}|;:,.<>?"; charset+=sp; req.append(secrets.choice(sp))
            rest = [secrets.choice(charset) for _ in range(length-len(req))]; pwd = req+rest; random.shuffle(pwd)
            password = "".join(pwd)
            return ToolResult(True, json.dumps({"password": password, "length": len(password), "entropy_bits": round(len(password)*6.5,1), "strength": "very_strong" if length>=16 else "strong" if length>=12 else "moderate"}))
        except Exception as e: return ToolResult(False, f"password_generate error: {e}")


class HashVerifyTool(BaseTool):
    name = "hash_verify"
    description = "Hash a password or verify against stored hash. Algorithms: bcrypt, sha256, sha512, md5."
    parameters = "password: str, hash_string: str = None, algorithm: str = 'sha256', mode: str = 'verify'"

    async def run(self, password: str, hash_string: str = None, algorithm: str = "sha256", mode: str = "verify") -> ToolResult:
        try:
            alg = algorithm.lower()
            if mode == "hash" or hash_string is None:
                if alg == "bcrypt":
                    bcrypt = _try_import("bcrypt")
                    if not bcrypt: return ToolResult(False, "Missing dep: pip install bcrypt")
                    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                elif alg in ("sha256","sha512","md5"): h = hashlib.new(alg, password.encode()).hexdigest()
                else: return ToolResult(False, f"Unsupported: {alg}")
                return ToolResult(True, json.dumps({"hash": h, "algorithm": alg, "mode": "generated"}))
            if alg == "bcrypt":
                bcrypt = _try_import("bcrypt")
                if not bcrypt: return ToolResult(False, "Missing dep: pip install bcrypt")
                valid = bcrypt.checkpw(password.encode(), hash_string.encode())
            elif alg in ("sha256","sha512","md5"): valid = secrets.compare_digest(hashlib.new(alg, password.encode()).hexdigest(), hash_string)
            else: return ToolResult(False, f"Unsupported: {alg}")
            return ToolResult(True, json.dumps({"valid": valid, "algorithm": alg}))
        except Exception as e: return ToolResult(False, f"hash_verify error: {e}")


class JwtDecodeTool(BaseTool):
    name = "jwt_decode"
    description = "Decode a JWT token. Returns header, payload, expiry status, signature validity."
    parameters = "token: str, secret_key: str = None, verify: bool = True"

    async def run(self, token: str, secret_key: str = None, verify: bool = True) -> ToolResult:
        try:
            parts = token.strip().split(".")
            if len(parts) != 3: return ToolResult(False, "Invalid JWT: need 3 parts")
            def b64d(s): s += "="*(4-len(s)%4); return json.loads(base64.urlsafe_b64decode(s))
            header = b64d(parts[0]); payload = b64d(parts[1])
            exp = payload.get("exp"); expired = bool(exp and exp < time.time()); sig_valid = None
            if verify and secret_key:
                jwt = _try_import("jwt")
                if jwt:
                    try: jwt.decode(token, secret_key, algorithms=[header.get("alg","HS256")]); sig_valid = True
                    except: sig_valid = False
            return ToolResult(True, json.dumps({"header": header, "payload": payload, "expired": expired, "signature_valid": sig_valid, "exp_datetime": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat() if exp else None, "valid": not expired and (sig_valid is not False)}, default=str))
        except Exception as e: return ToolResult(False, f"jwt_decode error: {e}")


class SslCheckTool(BaseTool):
    name = "ssl_check"
    description = "Check SSL certificate for a domain: expiry, issuer, days remaining, SAN list."
    parameters = "domain: str, check_expiry: bool = True, check_chain: bool = True"

    async def run(self, domain: str, check_expiry: bool = True, check_chain: bool = True) -> ToolResult:
        try:
            domain = re.sub(r"^https?://","",domain).split("/")[0].split(":")[0]
            ctx = ssl.create_default_context()
            def _get():
                with socket.create_connection((domain,443),timeout=10) as s:
                    with ctx.wrap_socket(s, server_hostname=domain) as ss: return ss.getpeercert()
            cert = await asyncio.get_event_loop().run_in_executor(None, _get)
            not_after = cert.get("notAfter","")
            dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z") if not_after else None
            days = (dt - datetime.utcnow()).days if dt else None
            issuer = next((v for t in cert.get("issuer",[]) for k,v in t if k=="organizationName"), None)
            return ToolResult(True, json.dumps({"domain": domain, "valid": days is None or days>0, "expired": days is not None and days<=0, "expiry_date": dt.isoformat() if dt else None, "days_remaining": days, "issuer": issuer, "chain_valid": True, "san": [v for t,v in cert.get("subjectAltName",[])]}, default=str))
        except ssl.SSLCertVerificationError as e: return ToolResult(False, json.dumps({"domain": domain, "valid": False, "error": str(e)}))
        except Exception as e: return ToolResult(False, f"ssl_check error: {e}")


class ApiKeyGenerateTool(BaseTool):
    name = "api_key_generate"
    description = "Generate secure API keys. Types: uuid, secret (hex), random (alphanumeric). Optional prefix."
    parameters = "key_type: str = 'uuid', length: int = 32, prefix: str = 'APIKEY_'"

    async def run(self, key_type: str = "uuid", length: int = 32, prefix: str = "APIKEY_") -> ToolResult:
        try:
            if key_type not in ["uuid","secret","random"]: return ToolResult(False, "key_type: uuid|secret|random")
            if not 8 <= length <= 128: return ToolResult(False, "length: 8–128")
            if key_type=="uuid": body = str(uuid.uuid4()).replace("-","")
            elif key_type=="secret": body = secrets.token_hex(length//2)
            else: body = "".join(secrets.choice(string.ascii_letters+string.digits) for _ in range(length))
            key = f"{prefix}{body}" if prefix else body
            return ToolResult(True, json.dumps({"api_key": key, "key_type": key_type, "length": len(key), "prefix": prefix}))
        except Exception as e: return ToolResult(False, f"api_key_generate error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# WEB & INTEGRATION (16–20)
# ══════════════════════════════════════════════════════════════════════════════

class SitemapParseTool(BaseTool):
    name = "sitemap_parse"
    description = "Fetch and parse sitemap.xml. Returns URL list with lastmod, changefreq, priority."
    parameters = "url: str, output_format: str = 'json'"

    async def run(self, url: str, output_format: str = "json") -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.get(url, headers={"User-Agent": "GazccAI/1.0"}); resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser"); urls = []
            for ut in soup.find_all("url"):
                e = {}
                for tag in ["loc","lastmod","changefreq","priority"]:
                    t = ut.find(tag)
                    if t: e[tag] = t.text.strip()
                if e.get("loc"): urls.append(e)
            for st in soup.find_all("sitemap"):
                loc = st.find("loc")
                if loc: urls.append({"loc": loc.text.strip(), "type": "sitemap_index"})
            if output_format == "list": return ToolResult(True, json.dumps([u.get("loc","") for u in urls]))
            return ToolResult(True, json.dumps({"source": url, "total_urls": len(urls), "urls": urls[:500]}))
        except Exception as e: return ToolResult(False, f"sitemap_parse error: {e}")


class WebhookTestTool(BaseTool):
    name = "webhook_test"
    description = "Send a test payload to a webhook URL. Returns status, response body, response time ms."
    parameters = "url: str, payload: dict, method: str = 'POST', headers: dict = None"

    async def run(self, url: str, payload: dict, method: str = "POST", headers: dict = None) -> ToolResult:
        try:
            h = {"Content-Type": "application/json", "User-Agent": "GazccAI-Webhook/1.0"}
            if headers: h.update(headers)
            t0 = time.time()
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.request(method.upper(), url, json=payload, headers=h)
            elapsed = round((time.time()-t0)*1000, 2)
            try: body = resp.json()
            except: body = resp.text[:1000]
            return ToolResult(resp.status_code<400, json.dumps({"status": resp.status_code, "body": body, "response_time_ms": elapsed, "success": resp.status_code<400}))
        except Exception as e: return ToolResult(False, f"webhook_test error: {e}")


class SeoAuditTool(BaseTool):
    name = "seo_audit"
    description = "SEO audit a URL: title, meta desc, h1, image alt, links, load time. Returns score + issues."
    parameters = "url: str, depth: int = 1"

    async def run(self, url: str, depth: int = 1) -> ToolResult:
        try:
            t0 = time.time()
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.get(url, headers={"User-Agent": "GazccAI-SEO/1.0"}); resp.raise_for_status()
            load_ms = round((time.time()-t0)*1000, 2)
            soup = BeautifulSoup(resp.text, "html.parser"); issues = []
            title = soup.find("title"); tt = title.text.strip() if title else ""
            if not tt: issues.append("Missing <title>")
            elif len(tt)>60: issues.append(f"Title too long ({len(tt)} chars)")
            elif len(tt)<30: issues.append(f"Title too short ({len(tt)} chars)")
            desc = soup.find("meta", attrs={"name":"description"}); dt = desc["content"] if desc and desc.get("content") else ""
            if not dt: issues.append("Missing meta description")
            elif len(dt)>160: issues.append(f"Meta desc too long ({len(dt)} chars)")
            h1s = soup.find_all("h1")
            if not h1s: issues.append("No <h1>")
            elif len(h1s)>1: issues.append(f"Multiple <h1> ({len(h1s)})")
            imgs = soup.find_all("img"); no_alt = [i for i in imgs if not i.get("alt")]
            if no_alt: issues.append(f"{len(no_alt)} images missing alt")
            links = soup.find_all("a", href=True)
            og = {t["property"]:t.get("content","") for t in soup.find_all("meta",property=True) if t["property"].startswith("og:")}
            return ToolResult(True, json.dumps({"url": url, "score": max(0,100-len(issues)*10), "load_time_ms": load_ms, "issues": issues, "meta_tags": {"title": tt, "description": dt, "og_tags": og}, "stats": {"h1": len(h1s), "h2": len(soup.find_all("h2")), "images": len(imgs), "links": len(links), "words": len(soup.get_text().split())}}))
        except Exception as e: return ToolResult(False, f"seo_audit error: {e}")


class OgPreviewTool(BaseTool):
    name = "og_preview"
    description = "Extract Open Graph + Twitter Card metadata from a URL. Optionally saves HTML preview file."
    parameters = "url: str, output_path: str = None"

    async def run(self, url: str, output_path: str = None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.get(url, headers={"User-Agent": "facebookexternalhit/1.1"}); resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            def gm(prop=None, name=None):
                t = soup.find("meta", property=prop) if prop else soup.find("meta", attrs={"name": name})
                return t["content"].strip() if t and t.get("content") else None
            og = {"url": url, "og_title": gm(prop="og:title") or (soup.find("title").text.strip() if soup.find("title") else None), "og_description": gm(prop="og:description") or gm(name="description"), "og_image": gm(prop="og:image"), "og_type": gm(prop="og:type"), "og_site_name": gm(prop="og:site_name"), "twitter_card": gm(name="twitter:card"), "twitter_image": gm(name="twitter:image")}
            if output_path:
                html = f'<!DOCTYPE html><html><head><title>OG Preview</title></head><body><h1>{og.get("og_title","")}</h1><p>{og.get("og_description","")}</p>{"<img src=" + repr(og["og_image"]) + " style=max-width:600px>" if og.get("og_image") else ""}</body></html>'
                Path(output_path).write_text(html); og["preview_html_path"] = output_path
            return ToolResult(True, json.dumps(og))
        except Exception as e: return ToolResult(False, f"og_preview error: {e}")


class LinkCheckerTool(BaseTool):
    name = "link_checker"
    description = "Crawl a URL and find all broken links (4xx/5xx). Returns broken list + summary."
    parameters = "url: str, timeout: int = 10, max_depth: int = 2"

    async def run(self, url: str, timeout: int = 10, max_depth: int = 2) -> ToolResult:
        try:
            visited, broken = set(), []; counts: Counter = Counter()
            queue = [(url, 0)]; base = re.sub(r"https?://([^/]+).*", r"\1", url)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                while queue and len(visited) < 50:
                    cur, depth = queue.pop(0)
                    if cur in visited: continue
                    visited.add(cur)
                    try:
                        r = await c.get(cur, headers={"User-Agent": "GazccAI-LinkChecker/1.0"})
                        counts[r.status_code] += 1
                        if r.status_code >= 400: broken.append({"url": cur, "status": r.status_code})
                        if depth < max_depth and "text/html" in r.headers.get("content-type",""):
                            s = BeautifulSoup(r.text, "html.parser")
                            for a in s.find_all("a", href=True):
                                h = a["href"]
                                if h.startswith("/"): h = f"https://{base}{h}"
                                if base in h and h not in visited: queue.append((h, depth+1))
                    except Exception as ex: broken.append({"url": cur, "status": "error", "error": str(ex)})
            return ToolResult(True, json.dumps({"base_url": url, "total_checked": len(visited), "broken_links": broken, "broken_count": len(broken), "status_codes": dict(counts), "summary": f"Checked {len(visited)} URLs, {len(broken)} broken"}))
        except Exception as e: return ToolResult(False, f"link_checker error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GAME & FUN (21–24)
# ══════════════════════════════════════════════════════════════════════════════

class RpgGeneratorTool(BaseTool):
    name = "rpg_generator"
    description = "Generate a random RPG character with stats, skills, equipment, HP, XP. Classes: warrior/mage/rogue/ranger/paladin."
    parameters = "character_class: str, level: int = 1, randomize: bool = True"

    async def run(self, character_class: str, level: int = 1, randomize: bool = True) -> ToolResult:
        try:
            CL = {
                "warrior": {"hp":120,"stats":{"str":16,"dex":10,"int":8,"con":14,"wis":10,"cha":10},"skills":["Slash","Shield Bash","Berserker Rage","Iron Skin","War Cry"],"gear":["Iron Sword","Steel Shield","Plate Armor"]},
                "mage":    {"hp":60, "stats":{"str":8,"dex":12,"int":18,"con":8,"wis":14,"cha":12},"skills":["Fireball","Ice Lance","Arcane Missile","Teleport","Mana Shield"],"gear":["Staff","Tome","Mage Robe"]},
                "rogue":   {"hp":80, "stats":{"str":12,"dex":18,"int":12,"con":10,"wis":10,"cha":14},"skills":["Backstab","Shadowstep","Poison Blade","Evasion","Pickpocket"],"gear":["Dagger","Shadow Cloak","Leather Armor"]},
                "ranger":  {"hp":90, "stats":{"str":12,"dex":16,"int":12,"con":12,"wis":14,"cha":10},"skills":["Arrow Rain","Hunter's Mark","Camouflage","Eagle Eye","Beast Call"],"gear":["Longbow","Quiver","Ranger Cloak"]},
                "paladin": {"hp":110,"stats":{"str":14,"dex":10,"int":10,"con":14,"wis":14,"cha":16},"skills":["Holy Strike","Divine Shield","Heal","Smite","Aura of Courage"],"gear":["Holy Sword","Tower Shield","Blessed Armor"]},
            }
            cls = character_class.lower()
            if cls not in CL: cls = random.choice(list(CL.keys()))
            d = CL[cls]; lvl = max(1, min(level, 100))
            stats = {k: v+(lvl-1)+(random.randint(0,3) if randomize else 1) for k,v in d["stats"].items()}
            hp = d["hp"]+lvl*10+(random.randint(-10,10) if randomize else 0)
            return ToolResult(True, json.dumps({"name": f"{cls.capitalize()} #{random.randint(1000,9999)}", "class": cls, "level": lvl, "hp": hp, "max_hp": hp, "xp": 0, "xp_to_next": lvl*1000, "stats": stats, "skills": random.sample(d["skills"], min(3+lvl//5,len(d["skills"]))), "equipment": d["gear"], "gold": lvl*random.randint(10,50)}))
        except Exception as e: return ToolResult(False, f"rpg_generator error: {e}")


class DiceRollerTool(BaseTool):
    name = "dice_roller"
    description = "Roll RPG dice (d4/d6/d8/d10/d12/d20/d100). Multiple dice + flat modifier. Detects crits."
    parameters = "dice_type: str = 'd20', count: int = 1, modifier: int = 0"

    async def run(self, dice_type: str = "d20", count: int = 1, modifier: int = 0) -> ToolResult:
        try:
            DICE = {"d4":4,"d6":6,"d8":8,"d10":10,"d12":12,"d20":20,"d100":100}
            dt = dice_type.lower()
            if dt not in DICE: return ToolResult(False, f"Invalid dice: {list(DICE.keys())}")
            if not 1 <= count <= 100: return ToolResult(False, "count: 1–100")
            rolls = [random.randint(1,DICE[dt]) for _ in range(count)]; total = sum(rolls)+modifier
            return ToolResult(True, json.dumps({"dice_type": dt, "count": count, "modifier": modifier, "rolls": rolls, "subtotal": sum(rolls), "total": total, "critical_hit": any(r==DICE[dt] for r in rolls) if DICE[dt]>=20 else False, "critical_fail": any(r==1 for r in rolls) if DICE[dt]>=20 else False}))
        except Exception as e: return ToolResult(False, f"dice_roller error: {e}")


class RandomQuestGenerateTool(BaseTool):
    name = "random_quest_generate"
    description = "Generate a random RPG quest: title, description, enemies, rewards, challenges."
    parameters = "difficulty: str = 'medium', theme: str = 'fantasy'"

    async def run(self, difficulty: str = "medium", theme: str = "fantasy") -> ToolResult:
        try:
            QP = {
                "fantasy": {"verbs":["Retrieve","Destroy","Rescue","Slay","Escort"],"objs":["the Ancient Artifact","the Dragon Egg","the Sacred Tome","the Lost Heir","the Dark Crystal"],"locs":["the Cursed Forest","Castle Darkmoore","Dragon's Peak","the Shadow Realm"],"enemies":["Shadow Wraiths","Dragon Cultists","Corrupted Knights","Demon Warlords"],"rewards":["Epic Weapon","Dragon Scale Armor","Ancient Spellbook","500 Gold"]},
                "sci-fi":  {"verbs":["Hack","Infiltrate","Extract","Disable"],"objs":["the AI Core","Classified Data","the Weapon Blueprint"],"locs":["the Space Station","the Corporate Megaplex"],"enemies":["Security Droids","Corporate Mercs","Rogue AI"],"rewards":["Advanced Exosuit","Hacking Module","50,000 Credits"]},
            }
            DD = {"easy":{"mult":1,"enemies":"1-2","time":"3 days"},"medium":{"mult":2,"enemies":"3-5","time":"7 days"},"hard":{"mult":4,"enemies":"8-12","time":"2 days"}}
            t = theme.lower() if theme.lower() in QP else "fantasy"; d = difficulty.lower() if difficulty.lower() in DD else "medium"
            qp = QP[t]; dd = DD[d]; verb=random.choice(qp["verbs"]); obj=random.choice(qp["objs"]); loc=random.choice(qp["locs"]); enemy=random.choice(qp["enemies"])
            return ToolResult(True, json.dumps({"title": f"{verb} {obj}", "difficulty": d, "theme": t, "description": f"You must {verb.lower()} {obj} from {loc}. {enemy} guard the path. Complete within {dd['time']}.", "enemies": enemy, "enemy_count": dd["enemies"], "rewards": random.sample(qp["rewards"],min(2,len(qp["rewards"]))), "xp_reward": 100*dd["mult"], "gold_reward": 50*dd["mult"], "time_limit": dd["time"], "challenges": [f"Defeat {enemy}", f"Navigate {loc}", f"Secure {obj}"]}))
        except Exception as e: return ToolResult(False, f"random_quest_generate error: {e}")


class PartyPlannerTool(BaseTool):
    name = "party_planner_guests"
    description = "Generate party checklist + budget breakdown for N guests. Themes: casual/formal/birthday/wedding."
    parameters = "count: int, budget: int = 1000000, theme: str = 'casual'"

    async def run(self, count: int, budget: int = 1000000, theme: str = "casual") -> ToolResult:
        try:
            if not 1 <= count <= 10000: return ToolResult(False, "count: 1–10,000")
            TH = {"casual":{"food":0.40,"drinks":0.20,"deco":0.15,"entertainment":0.10,"misc":0.15},"formal":{"food":0.35,"drinks":0.25,"deco":0.20,"entertainment":0.12,"misc":0.08},"birthday":{"food":0.30,"drinks":0.20,"deco":0.25,"entertainment":0.15,"misc":0.10},"wedding":{"food":0.40,"drinks":0.20,"deco":0.20,"entertainment":0.12,"misc":0.08}}
            t = theme.lower() if theme.lower() in TH else "casual"; p = TH[t]
            bd = {k: round(budget*v) for k,v in p.items()}
            recs = [f"Reserve venue for {int(count*1.1)} people","Send invites 2-3 weeks in advance"]
            if bd["food"]/count < 20000: recs.append("Food budget low — consider buffet")
            if count > 50: recs.append("50+ guests — hire catering")
            return ToolResult(True, json.dumps({"guest_count": count, "total_budget": budget, "per_person": round(budget/count), "theme": t, "budget_breakdown": bd, "recommendations": recs, "checklist": ["✓ Book venue","✓ Invitations","✓ Catering","✓ Decorate","✓ Entertainment","✓ Parking","✓ First aid","✓ Backup plan"]}))
        except Exception as e: return ToolResult(False, f"party_planner_guests error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING & EDUCATION (25–28)
# ══════════════════════════════════════════════════════════════════════════════

class QuizGenerateTool(BaseTool):
    name = "quiz_generate_from_text"
    description = "Generate quiz questions from text. Formats: multiple_choice, true_false, open."
    parameters = "text: str, question_count: int = 5, format: str = 'multiple_choice'"

    async def run(self, text: str, question_count: int = 5, format: str = "multiple_choice") -> ToolResult:
        try:
            if not text.strip(): return ToolResult(False, "text empty")
            question_count = max(1, min(question_count, 20))
            if format not in ["multiple_choice","true_false","open"]: format = "multiple_choice"
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip())>30]
            if not sents: return ToolResult(False, "Text too short")
            terms = list({w for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)*\b", text) if len(w)>3})[:15]
            questions = []
            for i in range(min(question_count, len(sents))):
                sent = sents[i]; hits = [t for t in terms if t in sent]; ans = hits[0] if hits else sent.split()[-2] if len(sent.split())>2 else "N/A"
                if format == "multiple_choice":
                    opts = [t for t in terms if t!=ans][:3]+[ans]
                    while len(opts)<4: opts.append(f"Option {len(opts)}")
                    random.shuffle(opts)
                    questions.append({"id":i+1,"question":f"Based on text: {sent[:100]}... Which is correct?","options":opts,"correct_answer":ans,"format":"multiple_choice"})
                elif format == "true_false":
                    questions.append({"id":i+1,"question":f"True or False: {sent}","correct_answer":"True","format":"true_false"})
                else:
                    questions.append({"id":i+1,"question":f"Explain: {sent[:100]}...","sample_answer":sent,"format":"open"})
            return ToolResult(True, json.dumps({"questions": questions, "total": len(questions), "format": format}))
        except Exception as e: return ToolResult(False, f"quiz_generate_from_text error: {e}")


class FlashcardGenerateTool(BaseTool):
    name = "flashcard_generate"
    description = "Generate study flashcards (front/back) from text. Saves to JSON."
    parameters = "text: str, cards_per_page: int = 5, output_path: str = 'flashcards.json'"

    async def run(self, text: str, cards_per_page: int = 5, output_path: str = "flashcards.json") -> ToolResult:
        try:
            if not text.strip(): return ToolResult(False, "text empty")
            cards = []
            for line in text.split("\n"):
                line = line.strip().lstrip("-*•").strip()
                if ":" in line and 10 < len(line) < 300:
                    f, b = line.split(":",1)
                    if f.strip() and b.strip(): cards.append({"id":len(cards)+1,"front":f.strip(),"back":b.strip()})
            if len(cards) < 3:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip())>20]
                for i,s in enumerate(sents):
                    w = s.split()
                    if len(w)>=5: cards.append({"id":i+1,"front":f"What is: {' '.join(w[:3])}?","back":" ".join(w[3:])})
            out = {"total_cards":len(cards),"pages":(len(cards)+cards_per_page-1)//cards_per_page,"cards":cards}
            if output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
            return ToolResult(True, json.dumps(out))
        except Exception as e: return ToolResult(False, f"flashcard_generate error: {e}")


class SummaryGenerateTool(BaseTool):
    name = "summary_generate"
    description = "Summarize text using extractive method. Styles: concise, detailed, bullet."
    parameters = "text: str, max_length: int = 500, style: str = 'concise'"

    async def run(self, text: str, max_length: int = 500, style: str = "concise") -> ToolResult:
        try:
            if not text.strip(): return ToolResult(False, "text empty")
            if style not in ["concise","detailed","bullet"]: style = "concise"
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip())>10]
            if not sents: return ToolResult(False, "No sentences found")
            sw = {"the","a","an","is","are","was","were","be","have","had","to","of","in","for","on","with","and","or","but","not","this","that","it","its","as","so"}
            freq = Counter(w for w in re.findall(r"\b\w+\b",text.lower()) if w not in sw and len(w)>3)
            scored = sorted([(sum(freq.get(w,0) for w in re.findall(r"\b\w+\b",s.lower()) if w not in sw), s) for s in sents], reverse=True)
            selected = []; budget = max_length
            for _,s in scored:
                if sum(len(x) for x in selected)+len(s) < budget: selected.append(s)
            ordered = [s for s in sents if s in selected]
            summary = "\n".join(f"• {s}" for s in ordered[:10]) if style=="bullet" else " ".join(ordered) if style=="detailed" else " ".join(ordered[:3])
            return ToolResult(True, json.dumps({"summary": summary[:max_length], "style": style, "original_length": len(text), "summary_length": len(summary), "compression_ratio": round(len(summary)/max(1,len(text)),3)}))
        except Exception as e: return ToolResult(False, f"summary_generate error: {e}")


class ConceptMapGenerateTool(BaseTool):
    name = "concept_map_generate"
    description = "Build a concept map JSON from text: nodes (concepts) + edges (relationships)."
    parameters = "text: str, output_path: str = None"

    async def run(self, text: str, output_path: str = None) -> ToolResult:
        try:
            if not text.strip(): return ToolResult(False, "text empty")
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip())>10]
            concepts = set()
            for s in sents:
                for c in re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]+)*\b", s):
                    if c not in {"The","A","An","In","On","At","Is","Are","This","That"}: concepts.add(c)
            sw = {"this","that","with","from","they","have","been","will","also"}
            freq = Counter(w for w in re.findall(r"\b\w{4,}\b",text.lower()) if w not in sw)
            concepts.update(w.capitalize() for w,_ in freq.most_common(10))
            cl = list(concepts)[:20]; nodes = [{"id":i,"label":c,"type":"concept"} for i,c in enumerate(cl)]
            edges = []; seen = set()
            for s in sents:
                present = [c for c in cl if c.lower() in s.lower()]
                for i in range(len(present)):
                    for j in range(i+1,min(i+3,len(present))):
                        a,b = cl.index(present[i]),cl.index(present[j]); key = (min(a,b),max(a,b))
                        if key not in seen: seen.add(key); rel = "causes" if any(w in s.lower() for w in ["causes","leads to"]) else "relates to"; edges.append({"source":a,"target":b,"relation":rel})
            out = {"nodes":nodes,"edges":edges[:50],"total_concepts":len(nodes),"total_relations":len(edges)}
            if output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(json.dumps(out,indent=2)); out["output_path"]=output_path
            return ToolResult(True, json.dumps(out))
        except Exception as e: return ToolResult(False, f"concept_map_generate error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DEVOPS & PERFORMANCE (29–32)
# ══════════════════════════════════════════════════════════════════════════════

class LoadTestTool(BaseTool):
    name = "load_test"
    description = "Lightweight HTTP load test: concurrent requests, avg/min/max ms, error rate, throughput."
    parameters = "url: str, concurrent_users: int = 10, duration: int = 10"

    async def run(self, url: str, concurrent_users: int = 10, duration: int = 10) -> ToolResult:
        try:
            concurrent_users = max(1,min(concurrent_users,50)); results=[]; errors=[]
            async def fire(c):
                try:
                    t0=time.time(); r=await c.get(url,timeout=10); results.append({"status":r.status_code,"time_ms":round((time.time()-t0)*1000,2)})
                except Exception as ex: errors.append(str(ex))
            async with httpx.AsyncClient(follow_redirects=True) as c:
                await asyncio.gather(*[fire(c) for _ in range(concurrent_users)], return_exceptions=True)
            times=[r["time_ms"] for r in results] or [0]
            return ToolResult(True, json.dumps({"url":url,"concurrent_users":concurrent_users,"total":len(results)+len(errors),"successful":len(results),"failed":len(errors),"error_rate_pct":round(len(errors)/max(len(results)+len(errors),1)*100,2),"avg_ms":round(sum(times)/len(times),2),"min_ms":min(times),"max_ms":max(times),"status_codes":dict(Counter(r["status"] for r in results)),"errors":errors[:5]}))
        except Exception as e: return ToolResult(False, f"load_test error: {e}")


class LogAnalyzerTool(BaseTool):
    name = "log_analyzer"
    description = "Analyze a log file: error frequency, top error lines, time range, recommendations."
    parameters = "file_path: str, error_patterns: list = None"

    async def run(self, file_path: str, error_patterns: list = None) -> ToolResult:
        try:
            p = Path(file_path)
            if not p.exists(): return ToolResult(False, f"Not found: {file_path}")
            if p.stat().st_size > 10*1024*1024: return ToolResult(False, "Log too large (>10MB)")
            content = p.read_text(errors="replace"); lines = content.splitlines()
            patterns = error_patterns or [r"\bERROR\b",r"\bCRITICAL\b",r"\bFATAL\b",r"\bException\b",r"\btimeout\b",r"\bfailed\b"]
            counts = {pat: len([l for l in lines if re.search(pat,l,re.IGNORECASE)]) for pat in patterns}
            err_lines = [l for l in lines if re.search(r"error|exception|fatal|critical",l,re.IGNORECASE)]
            ts = re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", content)
            recs = []
            if counts.get(r"\bERROR\b",0)>100: recs.append("High error rate — investigate immediately")
            if counts.get(r"\btimeout\b",0)>0: recs.append("Timeouts found — check service health")
            if counts.get(r"\bException\b",0)>0: recs.append("Unhandled exceptions — review error handling")
            if not recs: recs.append("Log appears healthy")
            return ToolResult(True, json.dumps({"file":file_path,"total_lines":len(lines),"error_summary":counts,"top_errors":list({l.strip() for l in err_lines})[:10],"time_range":{"first":ts[0],"last":ts[-1]} if len(ts)>=2 else {},"recommendations":recs,"stats":{"total_errors":sum(counts.values()),"error_rate_pct":round(len(err_lines)/max(len(lines),1)*100,2)}}))
        except Exception as e: return ToolResult(False, f"log_analyzer error: {e}")


class DbBackupTool(BaseTool):
    name = "db_backup"
    description = "Backup a SQLite database to .sql dump with all tables and INSERT statements."
    parameters = "db_path: str, backup_path: str = 'backup.sql', schedule: dict = None"

    async def run(self, db_path: str, backup_path: str = "backup.sql", schedule: dict = None) -> ToolResult:
        try:
            import sqlite3
            p = Path(db_path)
            if not p.exists(): return ToolResult(False, f"DB not found: {db_path}")
            conn = sqlite3.connect(db_path)
            lines = [f"-- GazccAI SQLite Backup — {datetime.now().isoformat()}\n-- Source: {db_path}\n\nPRAGMA foreign_keys=OFF;\nBEGIN TRANSACTION;\n"]
            for name,sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
                if not sql: continue
                lines.append(f"DROP TABLE IF EXISTS `{name}`;\n{sql};\n")
                rows = conn.execute(f"SELECT * FROM `{name}`").fetchall()
                for r in rows:
                    vals = ", ".join("NULL" if v is None else f"'{str(v).replace(chr(39),chr(39)*2)}'" if isinstance(v,str) else str(v) for v in r)
                    lines.append(f"INSERT INTO `{name}` VALUES ({vals});")
                lines.append("")
            lines.append("COMMIT;"); conn.close()
            dump = "\n".join(lines)
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True); Path(backup_path).write_text(dump)
            return ToolResult(True, json.dumps({"source":db_path,"backup_path":backup_path,"size_kb":round(len(dump)/1024,2),"created_at":datetime.now().isoformat(),"schedule":schedule or {}}))
        except Exception as e: return ToolResult(False, f"db_backup error: {e}")


class CacheCheckTool(BaseTool):
    name = "cache_check"
    description = "Check HTTP cache headers: Cache-Control, ETag, TTL, CDN status, cache effectiveness score (0-100)."
    parameters = "url: str, check_headers: bool = True, check_ttl: bool = True"

    async def run(self, url: str, check_headers: bool = True, check_ttl: bool = True) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                resp = await c.get(url, headers={"User-Agent": "GazccAI-CacheCheck/1.0"}); resp.raise_for_status()
            h = dict(resp.headers); cc=h.get("cache-control",""); etag=h.get("etag",""); lm=h.get("last-modified","")
            m = re.search(r"max-age=(\d+)", cc); ttl = int(m.group(1)) if m and check_ttl else None
            no_cache = "no-cache" in cc or "no-store" in cc
            score = (20 if etag else 0)+(10 if lm else 0)+(30 if ttl and ttl>3600 else 15 if ttl and ttl>0 else 0)+(20 if "public" in cc else 0)+(20 if any(h.get(k) for k in ["cf-cache-status","x-cache","x-varnish"]) else 0)
            return ToolResult(True, json.dumps({"url":url,"cache_status":"no-cache" if no_cache else "cacheable","is_cacheable":not no_cache and bool(ttl or etag),"ttl_seconds":ttl,"ttl_human":f"{ttl//3600}h {(ttl%3600)//60}m" if ttl else None,"cache_control":cc,"etag":etag,"last_modified":lm,"cdn_headers":{k:h.get(k,"") for k in ["cf-cache-status","x-cache","x-cache-status","via"] if h.get(k)},"cache_score":score,"valid":not no_cache,"headers":dict(list(h.items())[:20]) if check_headers else {}}))
        except Exception as e: return ToolResult(False, f"cache_check error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

PYTHON_AI_TOOLS = [
    ExcelParseTool, ApiTestTool, PdfMergeTool, QrCodeGenerateTool, PromptOptimizeTool,
    JsonSchemaGenerateTool, DataCleanTool, DataVizChartTool, CsvTransformTool, SqlGeneratorTool,
    PasswordGenerateTool, HashVerifyTool, JwtDecodeTool, SslCheckTool, ApiKeyGenerateTool,
    SitemapParseTool, WebhookTestTool, SeoAuditTool, OgPreviewTool, LinkCheckerTool,
    RpgGeneratorTool, DiceRollerTool, RandomQuestGenerateTool, PartyPlannerTool,
    QuizGenerateTool, FlashcardGenerateTool, SummaryGenerateTool, ConceptMapGenerateTool,
    LoadTestTool, LogAnalyzerTool, DbBackupTool, CacheCheckTool,
]


def register_python_ai_tools(registry, cfg: dict | None = None) -> None:
    """
    Register all 32 Python AI Tools into an existing ToolRegistry instance.

    USAGE in agent/core.py:
        from .python_ai_tools import register_python_ai_tools
        register_python_ai_tools(self._tools, self._cfg)
    """
    cfg = cfg or {}
    if not cfg.get("tools", {}).get("python_ai_tools", True):
        return
    for cls in PYTHON_AI_TOOLS:
        registry._register(cls())
