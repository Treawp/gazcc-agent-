"""
agent/fact_guard.py
══════════════════════════════════════════════════════════════════════════════
  GAZCC FACT GUARD — Anti-Hallucination Enforcement Layer
  ══════════════════════════════════════════════════════

  Dua komponen:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 1. FactGuardTool      — Tool yang bisa dipanggil agent secara eksplisit  │
  │    Actions: scan | verify | score                                         │
  │                                                                           │
  │ 2. FactGuardInterceptor — Auto-check SETIAP Final Answer di executor      │
  │    → Dipasang di executor.py, transparan, agent tidak perlu tahu          │
  └──────────────────────────────────────────────────────────────────────────┘

  MASALAH YANG DISELESAIKAN:
  - deep_reason/claude_reason bersifat voluntary → agent skip kalau "yakin"
  - Tidak ada verifikasi otomatis sebelum Final Answer dikembalikan ke user
  - Agent bisa hallucinate angka, versi, URL, nama fungsi, statistik

  CARA KERJA:
  1. Setiap Final Answer di-scan oleh FactGuardInterceptor (pure heuristic, 0 latency)
  2. Jika risk MEDIUM/HIGH → append ⚠️ FACT-CHECK block ke output
  3. Jika verify=True di config → auto-call web_search untuk klaim tertinggi
  4. FactGuardTool bisa dipanggil manual untuk scan/verify teks arbitrer

  INSTALL (core.py + executor.py):
    from agent.fact_guard import FactGuardTool, FactGuardInterceptor, register_fact_guard
    register_fact_guard(self._tools, self._cfg)
    self._fact_guard = FactGuardInterceptor(self._cfg, self._tools)

  REQUIREMENTS: tidak ada — pure stdlib + httpx (sudah ada)
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

try:
    from .tools import BaseTool, ToolResult
except ImportError:
    class ToolResult:
        def __init__(self, success: bool, output: str, metadata: dict | None = None):
            self.success = success
            self.output = output
            self.metadata = metadata or {}
        def __str__(self): return f"{'✓' if self.success else '✗'} {self.output}"
    class BaseTool:
        name = ""
        description = ""
        parameters = ""
        async def run(self, *a, **kw) -> ToolResult:
            return ToolResult(False, "not implemented")


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM PATTERNS — hal-hal yang sering di-hallucinate LLM
# ══════════════════════════════════════════════════════════════════════════════

CLAIM_PATTERNS: list[dict] = [
    # Versi software spesifik
    {
        "name":    "version_number",
        "pattern": r"\bv?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b",
        "risk":    "MEDIUM",
        "reason":  "Version numbers sering di-hallucinate — bisa aja versi itu gak exist",
    },
    # Statistik & persentase
    {
        "name":    "statistic",
        "pattern": r"\b\d+(?:\.\d+)?%|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:juta|ribu|billion|million|trillion)\b",
        "risk":    "HIGH",
        "reason":  "Angka statistik/persentase spesifik sering dikarang LLM",
    },
    # URL dan endpoint spesifik
    {
        "name":    "url_endpoint",
        "pattern": r"https?://[^\s\"'<>]+",
        "risk":    "MEDIUM",
        "reason":  "URL bisa mengarah ke halaman yang gak ada",
    },
    # Tanggal spesifik
    {
        "name":    "specific_date",
        "pattern": r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}\s+(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+\d{4}\b",
        "risk":    "HIGH",
        "reason":  "Tanggal spesifik rawan hallucination",
        "flags":   re.IGNORECASE,
    },
    # Nama fungsi/method/class di kode
    {
        "name":    "code_api_name",
        "pattern": r"`[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\([^)]{0,60}\)`",
        "risk":    "MEDIUM",
        "reason":  "Method chain spesifik bisa salah signature atau tidak exist",
    },
    # Klaim absolut yang over-confident
    {
        "name":    "absolute_claim",
        "pattern": r"\b(?:always|never|guaranteed|impossible|100%|definitely|certainly|absolutely|mustahil|pasti|dijamin|selalu|tidak pernah)\b",
        "risk":    "LOW",
        "reason":  "Klaim absolut tanpa konteks sering misleading",
        "flags":   re.IGNORECASE,
    },
    # Studi/penelitian tanpa kutipan
    {
        "name":    "uncited_study",
        "pattern": r"\b(?:studies show|research shows|according to (?:a )?(?:study|research|report)|penelitian menunjukkan|riset menunjukkan)\b",
        "risk":    "HIGH",
        "reason":  "Klaim 'penelitian menunjukkan' tanpa sumber = merah besar",
        "flags":   re.IGNORECASE,
    },
    # Nama orang + jabatan spesifik
    {
        "name":    "named_person_role",
        "pattern": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:is|was|sebagai|adalah|menjabat|CEO|CTO|founder|director|said|stated|mengatakan)\b",
        "risk":    "MEDIUM",
        "reason":  "Atribusi ke orang spesifik bisa salah atau dikarang",
    },
    # Package version di requirements/import
    {
        "name":    "package_pinned_version",
        "pattern": r"(?:pip install|npm install|yarn add|require|import)\s+[\w-]+(?:==|>=|<=|@)\d+[\w.]*",
        "risk":    "MEDIUM",
        "reason":  "Versi package yang di-pin bisa outdated atau tidak exist",
    },
]


@dataclass
class ClaimFlag:
    text: str
    claim_type: str
    risk: str           # LOW | MEDIUM | HIGH
    reason: str
    position: int
    verified: bool | None = None
    verify_note: str = ""


@dataclass
class ScanResult:
    risk_level: str             # OK | LOW | MEDIUM | HIGH
    risk_score: float           # 0.0 - 1.0
    flags: list[ClaimFlag] = field(default_factory=list)
    summary: str = ""
    text_length: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# CORE SCANNER — pure heuristic, zero latency, zero API call
# ══════════════════════════════════════════════════════════════════════════════

class HallucinationScanner:
    """
    Fast regex-based hallucination risk scanner.
    No API calls. Deterministic. ~0ms latency.
    """

    RISK_WEIGHTS = {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6}

    def scan(self, text: str) -> ScanResult:
        flags: list[ClaimFlag] = []

        for rule in CLAIM_PATTERNS:
            flags_val = rule.get("flags", 0)
            matches = re.finditer(rule["pattern"], text, flags_val)
            for m in matches:
                flags.append(ClaimFlag(
                    text       = m.group(0),
                    claim_type = rule["name"],
                    risk       = rule["risk"],
                    reason     = rule["reason"],
                    position   = m.start(),
                ))

        # Dedup: same text+type within 50 chars of each other
        seen: set[str] = set()
        unique_flags: list[ClaimFlag] = []
        for f in flags:
            key = f"{f.claim_type}::{f.text[:60]}"
            if key not in seen:
                seen.add(key)
                unique_flags.append(f)

        # Score: weighted sum, capped at 1.0
        raw_score = sum(self.RISK_WEIGHTS[f.risk] for f in unique_flags)
        score = min(raw_score / max(len(text) / 300, 1), 1.0)
        score = min(score + min(raw_score * 0.05, 0.5), 1.0)

        # Risk level
        if not unique_flags or score < 0.1:
            level = "OK"
        elif score < 0.3:
            level = "LOW"
        elif score < 0.6:
            level = "MEDIUM"
        else:
            level = "HIGH"

        # Summary
        by_type: dict[str, list[str]] = {}
        for f in unique_flags:
            by_type.setdefault(f.risk, []).append(f"{f.claim_type}({f.text[:40]})")

        summary_parts = []
        for risk in ("HIGH", "MEDIUM", "LOW"):
            if risk in by_type:
                summary_parts.append(f"{risk}: {', '.join(by_type[risk][:3])}")
        summary = " | ".join(summary_parts) if summary_parts else "No risky claims detected"

        return ScanResult(
            risk_level  = level,
            risk_score  = round(score, 3),
            flags       = unique_flags,
            summary     = summary,
            text_length = len(text),
        )

    def format_warning(self, result: ScanResult) -> str:
        if result.risk_level == "OK":
            return ""

        emoji = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}
        lines = [
            f"\n\n{'═'*60}",
            f"⚠️  FACT GUARD — Risk: {emoji.get(result.risk_level, '⚪')} {result.risk_level} (score={result.risk_score:.2f})",
            f"{'─'*60}",
            "Klaim berikut perlu diverifikasi sebelum digunakan:",
        ]

        shown = [f for f in result.flags if f.risk in ("HIGH", "MEDIUM")][:8]
        for i, flag in enumerate(shown, 1):
            verified_str = ""
            if flag.verified is True:
                verified_str = " ✅ VERIFIED"
            elif flag.verified is False:
                verified_str = f" ❌ SALAH — {flag.verify_note}"
            elif flag.verified is None and flag.risk == "HIGH":
                verified_str = " ❓ UNVERIFIED"

            lines.append(
                f"  [{i}] {flag.risk} | {flag.claim_type}: `{flag.text[:70]}`"
                + verified_str
            )
            lines.append(f"       └→ {flag.reason}")

        low_count = sum(1 for f in result.flags if f.risk == "LOW")
        if low_count:
            lines.append(f"  + {low_count} low-risk claim(s) tidak ditampilkan")

        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIER — web search check untuk klaim HIGH risk
# ══════════════════════════════════════════════════════════════════════════════

class ClaimVerifier:
    """
    Verifies specific claims using web search (DDG API).
    Only called for HIGH-risk flags or explicit verify action.
    """

    DDG_URL = "https://api.duckduckgo.com/"

    async def verify_claim(self, claim: str, context: str = "") -> tuple[bool | None, str]:
        """
        Returns (verified, note):
          True  → claim likely correct
          False → claim likely wrong
          None  → inconclusive / search failed
        """
        query = f"{claim} {context[:50]}".strip()
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    self.DDG_URL,
                    params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                    headers={"User-Agent": "GazccAgent/1.0"},
                )
                if r.status_code != 200:
                    return None, "Search unavailable"

                data = r.json()
                abstract = data.get("AbstractText", "") or data.get("Answer", "")
                if not abstract:
                    return None, "No direct result found — manual check recommended"

                # Simple heuristic: does the abstract mention our claim text?
                claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
                abstract_words = set(re.findall(r"\b\w{4,}\b", abstract.lower()))
                overlap = len(claim_words & abstract_words) / max(len(claim_words), 1)

                if overlap >= 0.5:
                    return True, abstract[:120]
                else:
                    return None, f"Result found but doesn't confirm claim: {abstract[:80]}"

        except Exception as e:
            return None, f"Verify failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# FACT GUARD TOOL — agent-callable tool
# ══════════════════════════════════════════════════════════════════════════════

class FactGuardTool(BaseTool):
    name = "fact_guard"
    description = (
        "Anti-hallucination scanner & verifier. "
        "Scans text for risky claims (numbers, versions, URLs, statistics, dates) "
        "and optionally verifies them via web search. "
        "Actions: "
        "scan (detect risky claims in text, no API call, instant), "
        "verify (verify a specific claim via web search), "
        "score (get just the risk score 0-1 for a text). "
        "Gunakan sebelum menyampaikan klaim faktual penting ke user."
    )
    parameters = (
        "action: str, "
        "text: str, "
        "claim: str = '', "
        "context: str = '', "
        "auto_verify_high: bool = False"
    )

    def __init__(self):
        self._scanner  = HallucinationScanner()
        self._verifier = ClaimVerifier()

    async def run(
        self,
        action: str,
        text: str,
        claim: str = "",
        context: str = "",
        auto_verify_high: bool = False,
    ) -> ToolResult:
        try:
            # ── score ──────────────────────────────────────────────────────
            if action == "score":
                result = self._scanner.scan(text)
                out = (
                    f"Risk Level : {result.risk_level}\n"
                    f"Risk Score : {result.risk_score:.3f}\n"
                    f"Flags Found: {len(result.flags)}\n"
                    f"Summary    : {result.summary}"
                )
                return ToolResult(True, out, {
                    "risk_level": result.risk_level,
                    "risk_score": result.risk_score,
                    "flag_count": len(result.flags),
                })

            # ── verify (single claim) ─────────────────────────────────────
            elif action == "verify":
                if not claim:
                    return ToolResult(False, "'claim' parameter wajib diisi untuk action verify.")
                verified, note = await self._verifier.verify_claim(claim, context)
                status = "✅ LIKELY CORRECT" if verified else ("❌ LIKELY WRONG" if verified is False else "❓ INCONCLUSIVE")
                out = (
                    f"Claim  : {claim}\n"
                    f"Result : {status}\n"
                    f"Note   : {note}"
                )
                return ToolResult(True, out, {"verified": verified, "note": note})

            # ── scan ───────────────────────────────────────────────────────
            elif action == "scan":
                result = self._scanner.scan(text)
                warning = self._scanner.format_warning(result)

                # Optional: auto-verify HIGH risk claims
                if auto_verify_high:
                    high_flags = [f for f in result.flags if f.risk == "HIGH"][:3]
                    for flag in high_flags:
                        verified, note = await self._verifier.verify_claim(flag.text, text[:100])
                        flag.verified   = verified
                        flag.verify_note = note
                    # Rebuild warning with verified info
                    warning = self._scanner.format_warning(result)

                if result.risk_level == "OK":
                    out = f"✅ Teks aman — tidak ada klaim berisiko terdeteksi.\nScore: {result.risk_score:.3f}"
                else:
                    out = (
                        f"Risk Level : {result.risk_level} (score={result.risk_score:.3f})\n"
                        f"Flags      : {len(result.flags)} klaim terdeteksi\n"
                        f"Summary    : {result.summary}"
                        + warning
                    )

                return ToolResult(
                    result.risk_level in ("OK", "LOW"),
                    out,
                    {
                        "risk_level": result.risk_level,
                        "risk_score": result.risk_score,
                        "flags": [
                            {"text": f.text, "type": f.claim_type, "risk": f.risk}
                            for f in result.flags[:10]
                        ],
                    },
                )

            else:
                return ToolResult(False, f"Unknown action: '{action}'. Valid: scan, verify, score.")

        except Exception as e:
            return ToolResult(False, f"fact_guard error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# FACT GUARD INTERCEPTOR — auto post-processor untuk executor
# ══════════════════════════════════════════════════════════════════════════════

class FactGuardInterceptor:
    """
    Auto-check setiap Final Answer sebelum dikembalikan ke user.
    Dipasang di executor.py — transparan, agent gak perlu panggil secara manual.

    Config keys (di config.yaml → tools:):
      fact_guard_enabled       : true
      fact_guard_min_length    : 100     # skip teks pendek < N chars
      fact_guard_auto_verify   : false   # auto web-search HIGH klaim (lebih lambat)
      fact_guard_block_threshold: 0.8    # score >= ini → tambah strong warning
    """

    def __init__(self, cfg: dict | None = None):
        self._cfg       = cfg or {}
        tool_cfg        = self._cfg.get("tools", {})
        self._enabled   = tool_cfg.get("fact_guard_enabled", True)
        self._min_len   = tool_cfg.get("fact_guard_min_length", 80)
        self._auto_verify = tool_cfg.get("fact_guard_auto_verify", False)
        self._block_thr = tool_cfg.get("fact_guard_block_threshold", 0.8)
        self._scanner   = HallucinationScanner()
        self._verifier  = ClaimVerifier()

    async def process(self, final_answer: str) -> str:
        """
        Call this on every Final Answer before returning to user.
        Returns (possibly annotated) final answer.
        """
        if not self._enabled:
            return final_answer

        if len(final_answer.strip()) < self._min_len:
            return final_answer  # Skip jawaban sangat pendek

        result = self._scanner.scan(final_answer)

        # OK or LOW → return as-is, no noise
        if result.risk_level in ("OK", "LOW"):
            return final_answer

        # MEDIUM → append soft warning
        if result.risk_level == "MEDIUM":
            warning = self._scanner.format_warning(result)
            return final_answer + warning

        # HIGH → auto-verify top flags if configured, then append strong warning
        if result.risk_level == "HIGH":
            if self._auto_verify:
                high_flags = [f for f in result.flags if f.risk == "HIGH"][:3]
                verify_tasks = [
                    self._verifier.verify_claim(f.text, final_answer[:150])
                    for f in high_flags
                ]
                verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)
                for flag, vr in zip(high_flags, verify_results):
                    if isinstance(vr, tuple):
                        flag.verified, flag.verify_note = vr

            warning = self._scanner.format_warning(result)

            # Extra strong header for very high risk
            if result.risk_score >= self._block_thr:
                header = (
                    "\n\n🚨 **PERINGATAN**: Respons ini mengandung klaim berisiko tinggi "
                    "yang mungkin tidak akurat. Verifikasi sebelum digunakan."
                )
                return final_answer + header + warning
            else:
                return final_answer + warning

        return final_answer

    def is_enabled(self) -> bool:
        return self._enabled


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def register_fact_guard(registry, cfg: dict | None = None) -> None:
    """
    Register FactGuardTool into ToolRegistry.
    Also call this to get the FactGuardInterceptor instance.

    USAGE in agent/core.py:
        from agent.fact_guard import register_fact_guard, FactGuardInterceptor
        register_fact_guard(self._tools, self._cfg)
        self._fact_guard = FactGuardInterceptor(self._cfg)
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})
    if tool_cfg.get("fact_guard_enabled", True):
        registry.register(FactGuardTool())
