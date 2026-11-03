"""전체 흐름: 변환 → 청킹 → 1단계 판정 → 2단계 청크 추출 → (이미지 보완) → 3단계 통합 → 검증 → 산출물.

같은 모델을 순서대로 호출하는 단순한 Python 흐름이다(에이전트 간 대화 없음).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .chunking import estimate_tokens, plan_chunks, render_block
from .config import PROMPT_VERSION, SCHEMA_VERSION, Config
from .llm import BudgetedLLM, BudgetExceeded, CallBudget, LLMClient, LLMError
from .models import (
    Block,
    Chunk,
    ChunkExtraction,
    Document,
    ImageExtraction,
    MergeOutput,
    PageTranscript,
    PaperGraph,
    ReviewItem,
    TypeDecision,
)
from .pdf_to_md import convert_pdf, render_page_png, to_markdown_text
from .validation import validate_and_finalize

Log = Callable[[str], None]


@dataclass
class PipelineResult:
    workdir: Path
    graph: PaperGraph | None
    partial_reasons: list[str] = field(default_factory=list)
    message: str = ""


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return s[:40] or "paper"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump_json"):
        path.write_text(data.model_dump_json(indent=2), "utf-8")
    else:
        def enc(o):
            if hasattr(o, "model_dump"):
                return o.model_dump()
            return str(o)

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=enc), "utf-8")


def sanitize_extraction(ext: ChunkExtraction, valid_ids: set[str]) -> int:
    """청크 추출 결과에서 존재하지 않는 근거 ID를 제거하고 제거 수를 반환한다."""
    removed = 0

    def clean(ids: list[str]) -> list[str]:
        nonlocal removed
        good = [i for i in ids if i in valid_ids]
        removed += len(ids) - len(good)
        return good

    for e in ext.entities:
        e.evidence_block_ids = clean(e.evidence_block_ids)
    for f in ext.facts:
        f.evidence_block_ids = clean(f.evidence_block_ids)
    for r in ext.relations:
        r.evidence_block_ids = clean(r.evidence_block_ids)
    for u in ext.unresolved:
        u.block_ids = clean(u.block_ids)
    ext.readability_block_ids = clean(ext.readability_block_ids)
    return removed


class Pipeline:
    def __init__(self, cfg: Config, client: LLMClient | None, log: Log = print):
        cfg.validate()
        self.cfg = cfg
        self.client = client
        self.log = log

    # ------------------------------------------------------------------
    def plan(self, doc: Document) -> tuple[list[Chunk], int, list[str]]:
        cfg = self.cfg
        slots = cfg.max_api_calls - 2 - cfg.image_reserve  # 판정 1 + 통합 1 + 이미지 예약
        chunks, _, _, notes = plan_chunks(
            doc, cfg.chunk_target_tokens, cfg.chunk_max_tokens, cfg.chunk_hard_max_tokens, max(slots, 1)
        )
        return chunks, max(slots, 1), notes

    def run(self, pdf_path: Path, dry_run: bool = False) -> PipelineResult:
        cfg, log = self.cfg, self.log
        pdf_path = Path(pdf_path)
        doc = convert_pdf(pdf_path, cfg.min_chars_per_page, cfg.max_garbled_ratio)
        paper_id = f"{slugify(pdf_path.stem)}-{doc.pdf_sha256[:8]}"
        workdir = cfg.output_dir / paper_id
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "paper.md").write_text(to_markdown_text(doc), "utf-8")
        _write_json(workdir / "blocks.json", doc)
        for w in doc.warnings:
            log(f"[경고] {w}")

        chunks, slots, plan_notes = self.plan(doc)
        for n in plan_notes:
            log(f"[계획] {n}")
        # 판독 이상 페이지: 손상 문자가 많은 순. 이미지 보완은 이 중 최대 max_image_pages개만 한다.
        code_flag_pages = [
            p.page for p in sorted(doc.pages, key=lambda p: (-p.garbled_chars, p.page))
            if p.readability_flags and not p.is_reference_only
        ]
        expected = 2 + min(len(chunks), slots) + min(len(code_flag_pages), cfg.max_image_pages)
        log(
            f"[계획] 청크 {len(chunks)}개(추출 슬롯 {slots}), 코드 판독 이상 페이지 {code_flag_pages or '없음'} "
            f"→ 예상 호출 {expected}회 / 한도 {cfg.max_api_calls}회"
        )
        _write_json(workdir / "chunks.json", chunks)
        if dry_run:
            return PipelineResult(workdir, None, message="dry-run: API 호출 없이 계획만 저장했다.")
        if self.client is None:
            raise ValueError("LLM 클라이언트가 없다(dry-run이 아니면 필요).")

        budget = CallBudget(cfg.max_api_calls)
        llm = BudgetedLLM(
            self.client, budget, workdir / "cache",
            {**cfg.cache_fingerprint(), "pdf_sha256": doc.pdf_sha256, "converter": doc.converter},
            cfg.model,
        )
        partial_reasons: list[str] = []
        review: list[ReviewItem] = []
        notes: list[str] = []  # 통합 호출에 전달할 처리 상태 메모
        notes.extend(doc.warnings)
        notes.extend(plan_notes)
        image_pages_used: list[int] = []

        def finish_usage() -> None:
            _write_json(workdir / "usage.json", {"totals": budget.totals(), "records": [r.__dict__ for r in budget.records]})

        try:
            # --- 1단계 -------------------------------------------------
            page1 = doc.pages[0]
            if page1.readability_flags and not page1.is_reference_only and cfg.max_image_pages > 0:
                self._transcribe_first_page(llm, doc, pdf_path, image_pages_used, notes)
            decision = self._classify(llm, doc)
            _write_json(workdir / "classification.json", decision)
            log(f"[1단계] 연구 유형: {decision.study_type} — {decision.design_description[:80]}")
            decision_json = decision.model_dump_json()

            # --- 2단계 -------------------------------------------------
            valid_ids = {b.id for b in doc.blocks}
            partials: dict[str, ChunkExtraction] = {}
            todo = list(chunks)
            first, deferred = todo[:slots], todo[slots:]
            image_needed = lambda: max(0, min(len(code_flag_pages), cfg.max_image_pages) - len(image_pages_used))  # noqa: E731
            self._extract_chunks(llm, first, decision, decision_json, len(chunks), partials, valid_ids,
                                 workdir, review, partial_reasons, budget, reserve_extra=image_needed, doc=doc)

            # --- 이미지 보완 ------------------------------------------------
            self._image_supplement(llm, doc, pdf_path, partials, decision_json, image_pages_used, code_flag_pages,
                                   valid_ids, workdir, review, notes, budget)

            # --- 여유 예산으로 남은 청크 처리 -------------------------------------
            if deferred:
                self._extract_chunks(llm, deferred, decision, decision_json, len(chunks), partials, valid_ids,
                                     workdir, review, partial_reasons, budget, reserve_extra=lambda: 0, doc=doc,
                                     leftover_mode=True)

            unprocessed = [c.id for c in chunks if c.status != "done"]
            if unprocessed:
                partial_reasons.append(f"처리하지 못한 청크: {unprocessed}")
            _write_json(workdir / "chunks.json", chunks)

            # --- 3단계 -------------------------------------------------
            graph = self._merge(llm, doc, decision_json, chunks, partials, notes, partial_reasons, review,
                                paper_id, workdir)
            finish_usage()
            if graph is None:
                return PipelineResult(workdir, None, partial_reasons, "통합 보류: 부분 추출 결과만 저장했다.")
            _write_json(workdir / "graph.json", graph)
            _write_json(workdir / "review.json", graph.review_items)
            self._write_manifest(workdir, doc, chunks, partial_reasons, budget)
            return PipelineResult(workdir, graph, partial_reasons, "완료(사람 검토 전)" if graph.status.complete else "부분 완료(사람 검토 전)")
        except (LLMError, BudgetExceeded) as e:
            finish_usage()
            self._write_manifest(workdir, doc, chunks, partial_reasons + [f"중단: {e}"], budget)
            return PipelineResult(workdir, None, partial_reasons + [str(e)], f"중단: {e} (완료된 부분은 캐시에 보존됨)")

    # ------------------------------------------------------------------
    def _page_text(self, doc: Document, page: int) -> str:
        return "\n\n".join(f"[{b.id}]\n{render_block(b)}" for b in doc.page_blocks(page) if not b.excluded)

    def _transcribe_first_page(self, llm: BudgetedLLM, doc: Document, pdf: Path, used: list[int], notes: list[str]) -> None:
        """첫 페이지가 판독 불가이면 판정 전에 이미지로 보완한다(참고문헌 페이지가 아님을 확인한 경우)."""
        self.log("[이미지] 첫 페이지 판독 이상 → 판정 전 이미지 보완")
        out = llm.run(
            stage="transcribe-p1", system=prompts.TRANSCRIBE_SYSTEM, user=prompts.transcribe_user(1, self._page_text(doc, 1)),
            schema=PageTranscript, max_output_tokens=self.cfg.max_output_tokens_image,
            images=[render_page_png(pdf, 1)], retries=self.cfg.max_retries_per_call, keep_free=2,
        )
        self._add_image_block(doc, 1, out.transcript)
        used.append(1)
        notes.append("1페이지는 이미지 판독문으로 보완됨(자동 추출 결과, 원본 텍스트 아님)")

    @staticmethod
    def _add_image_block(doc: Document, page: int, text: str) -> Block:
        b = Block(id=f"p{page}-img", page=page, kind="image_transcript", text=text, source="image")
        doc.blocks = [x for x in doc.blocks if x.id != b.id] + [b]
        return b

    def _classify(self, llm: BudgetedLLM, doc: Document) -> TypeDecision:
        text = self._page_text(doc, 1)
        return llm.run(
            stage="classify", system=prompts.CLASSIFY_SYSTEM, user=prompts.classify_user(text),
            schema=TypeDecision, max_output_tokens=self.cfg.max_output_tokens_classify,
            retries=self.cfg.max_retries_per_call, keep_free=1,
        )

    def _extract_chunks(self, llm, chunks, decision, decision_json, total, partials, valid_ids, workdir, review,
                        partial_reasons, budget, reserve_extra, doc, leftover_mode: bool = False) -> None:
        cfg = self.cfg
        system = prompts.extract_system(decision.study_type)
        for i, ch in enumerate(chunks):
            remaining_after = len(chunks) - i - 1
            if leftover_mode and budget.remaining - 1 < 1:  # 통합 호출 1회 예약
                ch.status = "skipped_budget"
                continue
            try:
                ext = llm.run(
                    stage=f"extract-{ch.id}", system=system,
                    user=prompts.extract_user(ch, decision_json, total, int(ch.id[1:])),
                    schema=ChunkExtraction, max_output_tokens=cfg.max_output_tokens_extract,
                    retries=cfg.max_retries_per_call,
                    keep_free=1 + (0 if leftover_mode else remaining_after) + reserve_extra(),
                )
            except BudgetExceeded as e:
                ch.status = "skipped_budget"
                self.log(f"[2단계] {ch.id} 예산 부족: {e}")
                continue
            except LLMError as e:
                ch.status = "failed"
                partial_reasons.append(f"{ch.id} 추출 실패: {e}")
                self.log(f"[2단계] {ch.id} 실패: {e}")
                continue
            removed = sanitize_extraction(ext, valid_ids)
            if removed:
                review.append(ReviewItem(kind="invalid_evidence", description=f"{ch.id}: 존재하지 않는 근거 ID {removed}개 제거", related_ids=[ch.id]))
            partials[ch.id] = ext
            ch.status = "done"
            _write_json(workdir / "extractions" / f"{ch.id}.json", ext)
            self.log(f"[2단계] {ch.id} 완료 (사실 {len(ext.facts)}, 관계 단서 {len(ext.relations)}, 미해결 {len(ext.unresolved)})")

    def _image_supplement(self, llm, doc, pdf, partials, decision_json, used, code_pages, valid_ids, workdir, review,
                          notes, budget) -> None:
        cfg = self.cfg
        issues: dict[int, list[str]] = {}
        for p in code_pages:
            issues.setdefault(p, []).append("코드 판독 이상 신호")
        bmap = doc.block_map()
        for cid, ext in partials.items():
            for bid in ext.readability_block_ids:
                b = bmap.get(bid)
                if b and not b.excluded:
                    issues.setdefault(b.page, []).append(f"LLM 판독 문제 지적({cid}:{bid})")
        # 참고문헌 전용 페이지 제외, 이미 보완한 페이지 제외
        info = {p.page: p for p in doc.pages}
        cands = [p for p in issues if not info[p].is_reference_only and p not in used]
        cands.sort(key=lambda p: (0 if any(i.startswith("코드") for i in issues[p]) else 1, -info[p].garbled_chars, p))
        for page in cands:
            if len(used) >= cfg.max_image_pages:
                review.append(ReviewItem(kind="unreadable", description=f"p{page}: 이미지 보완 한도({cfg.max_image_pages}페이지) 초과로 보완하지 못함", related_ids=[f"p{page}"]))
                notes.append(f"p{page}는 판독 문제가 있으나 이미지 보완하지 못함 → 관련 항목을 미보고로 단정하지 말 것")
                continue
            if budget.remaining - 1 < 1:
                review.append(ReviewItem(kind="unreadable", description=f"p{page}: 호출 예산 부족으로 이미지 보완하지 못함", related_ids=[f"p{page}"]))
                notes.append(f"p{page}는 판독 문제가 있으나 호출 예산 부족으로 보완하지 못함")
                continue
            block_id = f"p{page}-img"
            try:
                out = llm.run(
                    stage=f"image-p{page}", system=prompts.extract_system(json.loads(decision_json)["study_type"]) + prompts.IMAGE_SYSTEM_SUFFIX,
                    user=prompts.image_user(page, block_id, issues[page], self._page_text(doc, page), decision_json),
                    schema=ImageExtraction, max_output_tokens=cfg.max_output_tokens_image,
                    images=[render_page_png(pdf, page)], retries=0, keep_free=1,
                )
            except (LLMError, BudgetExceeded) as e:
                review.append(ReviewItem(kind="unreadable", description=f"p{page}: 이미지 보완 실패 — {e}", related_ids=[f"p{page}"]))
                notes.append(f"p{page} 이미지 보완 실패 → 관련 항목을 미보고로 단정하지 말 것")
                continue
            used.append(page)
            self._add_image_block(doc, page, out.transcript)
            valid_ids.add(block_id)
            sanitize_extraction(out.extraction, valid_ids)
            partials[f"img-p{page}"] = out.extraction
            _write_json(workdir / "extractions" / f"img-p{page}.json", out)
            notes.append(f"p{page}는 이미지 판독문({block_id})으로 보완됨(자동 추출 결과)")
            self.log(f"[이미지] p{page} 보완 완료")
        _write_json(workdir / "blocks.json", doc)
        (workdir / "paper.md").write_text(to_markdown_text(doc), "utf-8")

    def _merge(self, llm, doc, decision_json, chunks, partials, notes, partial_reasons, review, paper_id, workdir):
        cfg = self.cfg
        bmap = doc.block_map()
        payload = [
            {"chunk_id": cid, "heading_path": next((c.heading_path for c in chunks if c.id == cid), []),
             "extraction": json.loads(ext.model_dump_json())}
            for cid, ext in partials.items()
        ]
        partials_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        # 유형 충돌·모순 근거 블록 발췌 (한도를 넘는 블록은 조용히 자르지 않고 검토 항목으로)
        want: list[str] = []
        for ext in partials.values():
            for u in ext.unresolved:
                if u.kind in ("type_conflict", "contradiction"):
                    want.extend(u.block_ids)
        excerpts, used_tokens = [], 0
        for bid in dict.fromkeys(want):
            b = bmap.get(bid)
            if not b:
                continue
            t = estimate_tokens(b.text)
            if used_tokens + t > cfg.merge_excerpt_max_tokens:
                review.append(ReviewItem(kind="unprocessed", description=f"{bid}: 발췌 한도 초과로 통합 입력에 포함하지 못함", evidence_block_ids=[bid]))
                continue
            excerpts.append(f"[{bid}]\n{b.text}")
            used_tokens += t

        user = prompts.merge_user(decision_json, partials_json, "\n\n".join(excerpts), notes + partial_reasons)
        total = estimate_tokens(prompts.MERGE_SYSTEM) + estimate_tokens(user)
        if total > cfg.merge_max_input_tokens:
            msg = f"통합 입력 약 {total}토큰이 한도 {cfg.merge_max_input_tokens}를 넘어 통합을 보류했다(조용히 자르지 않음)."
            partial_reasons.append(msg)
            self.log(f"[3단계] {msg}")
            _write_json(workdir / "partials.json", payload)
            return None
        merged = llm.run(
            stage="merge", system=prompts.MERGE_SYSTEM, user=user, schema=MergeOutput,
            max_output_tokens=cfg.max_output_tokens_merge, retries=cfg.max_retries_per_call, keep_free=0,
        )
        self.log(f"[3단계] 통합 완료: 유형 {merged.final_study_type}, 집단 {len(merged.populations)}, 프로그램 {len(merged.programs)}, 결과 {len(merged.outcomes)}, 관계 {len(merged.relations)}")
        return validate_and_finalize(
            merged, doc, paper_id=paper_id, model=cfg.model, schema_version=SCHEMA_VERSION,
            prompt_version=PROMPT_VERSION, partial_reasons=partial_reasons, extra_review=review,
        )

    def _write_manifest(self, workdir, doc, chunks, partial_reasons, budget) -> None:
        _write_json(workdir / "run_manifest.json", {
            "pdf_name": doc.pdf_name, "pdf_sha256": doc.pdf_sha256, "converter": doc.converter,
            "config": self.cfg.to_dict(), "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
            "chunk_status": {c.id: c.status for c in chunks}, "partial_reasons": partial_reasons,
            "usage": budget.totals(), "conversion_warnings": doc.warnings,
        })

