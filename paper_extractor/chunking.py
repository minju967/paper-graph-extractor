"""준비 B: 제목 계층 기반 청킹.

코드는 의학 정보를 미리 찾지 않는다. 제목 경로와 원문 순서를 유지하며 입력 크기만 조절한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Block, Chunk, ChunkPart, Document
from .pdf_to_md import clean_heading

_CAPTION_RE = re.compile(r"^\s*\**\s*(table|표)\s*\d+", re.IGNORECASE)
_TABLE_NOTE_RE = re.compile(r"^\s*\**\s*(note|notes|주|비고|abbreviations?|\*|†|‡|a\)|\(a\))", re.IGNORECASE)

_enc = None
_enc_tried = False


def estimate_tokens(text: str) -> int:
    """토큰 수 추정. tiktoken을 쓸 수 없으면 문자 기반 보수적 추정으로 대체한다."""
    global _enc, _enc_tried
    if not _enc_tried:
        _enc_tried = True
        try:
            import tiktoken

            _enc = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - 오프라인 등
            _enc = None
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:  # noqa: BLE001
            pass
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return int(ascii_chars / 3.5 + (len(text) - ascii_chars) / 1.3) + 1


def render_block(b: Block) -> str:
    if b.kind == "heading":
        return "#" * (b.heading_level or 2) + " " + b.text
    return b.text


@dataclass
class _Atom:
    parts: list[ChunkPart]
    tokens: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Section:
    path: list[str]
    atoms: list[_Atom] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(a.tokens for a in self.atoms)


def _split_text(text: str, limit: int) -> list[str]:
    """긴 문단을 문장 경계로 나눈다(경계가 없으면 글자 수 기준)."""
    sentences = re.split(r"(?<=[.!?。다요])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if cur and estimate_tokens(cur + " " + s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    final: list[str] = []
    for piece in out:  # 문장 하나가 한도를 넘는 극단적 경우
        while estimate_tokens(piece) > limit and len(piece) > 200:
            cut = max(200, int(len(piece) * limit / max(estimate_tokens(piece), 1)))
            final.append(piece[:cut])
            piece = piece[cut:]
        final.append(piece)
    return final


def _table_atoms(table: Block, caption: Block | None, note: Block | None, limit: int) -> list[_Atom]:
    """표는 제목·열 이름·각주를 함께 둔다. 한도를 넘으면 행 단위로 나누고 머리말을 반복한다."""
    cap_text = caption.text if caption else ""
    note_text = note.text if note else ""
    lines = table.text.splitlines()
    header = lines[:2]  # 열 이름 + 구분선
    rows = lines[2:]
    full = "\n".join(filter(None, [cap_text, table.text, note_text]))
    if estimate_tokens(full) <= limit or len(rows) < 2:
        parts = []
        if caption:
            parts.append(ChunkPart(block_id=caption.id, text=caption.text))
        parts.append(ChunkPart(block_id=table.id, text=table.text))
        if note:
            parts.append(ChunkPart(block_id=note.id, text=note.text))
        w = [] if estimate_tokens(full) <= limit else [f"{table.id}: 나눌 수 없는 표가 청크 한도를 넘는다"]
        return [_Atom(parts, estimate_tokens(full), w)]

    atoms: list[_Atom] = []
    base = estimate_tokens("\n".join(filter(None, [cap_text, "\n".join(header), note_text])))
    cur: list[str] = []
    cur_tokens = base
    groups: list[list[str]] = []
    for r in rows:
        t = estimate_tokens(r)
        if cur and cur_tokens + t > limit:
            groups.append(cur)
            cur, cur_tokens = [], base
        cur.append(r)
        cur_tokens += t
    if cur:
        groups.append(cur)
    for gi, g in enumerate(groups):
        body = "\n".join(header + g)
        parts = []
        if caption:  # 공통 설명(제목)을 각 부분에 포함
            parts.append(ChunkPart(block_id=caption.id, text=caption.text, is_table_continuation=gi > 0))
        parts.append(ChunkPart(block_id=table.id, text=body, is_table_continuation=gi > 0))
        if note:
            parts.append(ChunkPart(block_id=note.id, text=note.text, is_table_continuation=gi > 0))
        atoms.append(
            _Atom(parts, estimate_tokens("\n".join(p.text for p in parts)), [f"{table.id}: 긴 표를 {len(groups)}부분으로 나눔({gi + 1}/{len(groups)})"])
        )
    return atoms


def _block_atoms(blocks: list[Block], limit: int) -> list[_Atom]:
    atoms: list[_Atom] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind == "table":
            cap = None
            if atoms and atoms[-1].parts and len(atoms[-1].parts) == 1 and _CAPTION_RE.match(atoms[-1].parts[0].text):
                prev = atoms.pop()
                cap = Block(id=prev.parts[0].block_id, page=b.page, kind="paragraph", text=prev.parts[0].text)
            note = None
            if i + 1 < len(blocks) and blocks[i + 1].kind == "paragraph" and _TABLE_NOTE_RE.match(blocks[i + 1].text):
                note = blocks[i + 1]
                i += 1
            atoms.extend(_table_atoms(b, cap, note, limit))
        else:
            text = render_block(b)
            tok = estimate_tokens(text)
            if tok > limit and b.kind == "paragraph":
                pieces = _split_text(text, limit)
                for k, piece in enumerate(pieces):
                    atoms.append(
                        _Atom(
                            [ChunkPart(block_id=b.id, text=piece, is_table_continuation=False)],
                            estimate_tokens(piece),
                            [f"{b.id}: 긴 문단을 문장 경계로 나눔({k + 1}/{len(pieces)})"],
                        )
                    )
            else:
                atoms.append(_Atom([ChunkPart(block_id=b.id, text=text)], tok))
        i += 1
    return atoms


def build_sections(doc: Document, limit: int) -> tuple[list[_Section], list[str]]:
    warnings: list[str] = []
    blocks = [b for b in doc.blocks if not b.excluded]
    has_headings = any(b.kind == "heading" for b in blocks)
    if not has_headings:
        warnings.append("제목 인식 실패: 페이지 경계로 구역을 나눈다.")

    groups: list[tuple[list[str], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    cur_path: list[str] = []
    cur_blocks: list[Block] = []
    last_page = None

    def push() -> None:
        nonlocal cur_blocks
        if cur_blocks:
            groups.append((cur_path, cur_blocks))
        cur_blocks = []

    for b in blocks:
        if not has_headings and last_page is not None and b.page != last_page:
            push()
        last_page = b.page
        if b.kind == "heading":
            push()
            lvl = b.heading_level or 2
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, clean_heading(b.text)))
            cur_path = [t for _, t in stack]
        cur_blocks.append(b)
    push()

    sections = []
    for path, bl in groups:
        sec = _Section(path=list(path), atoms=_block_atoms(bl, limit))
        sections.append(sec)
    return sections, warnings


def _common_prefix(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []
    out = []
    for items in zip(*paths):
        if all(x == items[0] for x in items):
            out.append(items[0])
        else:
            break
    return out


def chunk_document(doc: Document, target_tokens: int, max_tokens: int) -> list[Chunk]:
    sections, doc_warnings = build_sections(doc, max_tokens)
    chunks: list[Chunk] = []
    cur_atoms: list[_Atom] = []
    cur_paths: list[list[str]] = []
    cur_tokens = 0

    def emit() -> None:
        nonlocal cur_atoms, cur_paths, cur_tokens
        if not cur_atoms:
            return
        parts = [p for a in cur_atoms for p in a.parts]
        warns = [w for a in cur_atoms for w in a.warnings]
        if doc_warnings:
            warns = doc_warnings + warns
        chunks.append(
            Chunk(
                id=f"c{len(chunks) + 1:02d}",
                heading_path=_common_prefix(cur_paths),
                parts=parts,
                token_estimate=cur_tokens,
                warnings=warns,
            )
        )
        cur_atoms, cur_paths, cur_tokens = [], [], 0

    for sec in sections:
        sec_tokens = sec.tokens
        # 짧은 인접 절은 각자의 제목 블록을 유지한 채 목표 크기까지 묶는다.
        # (PyMuPDF4LLM은 제목 수준을 거의 구분하지 못해 '같은 상위 절' 조건을 걸면 거의 묶이지 않는다.)
        if cur_atoms and cur_tokens + sec_tokens > target_tokens:
            emit()
        if sec_tokens <= target_tokens or not cur_atoms and sec_tokens <= max_tokens:
            cur_atoms.extend(sec.atoms)
            cur_paths.append(sec.path)
            cur_tokens += sec_tokens
            continue
        # 긴 절: 원자(문단·표 부분) 경계로 목표 크기까지 채운다.
        for atom in sec.atoms:
            if cur_atoms and cur_tokens + atom.tokens > target_tokens:
                emit()
            cur_atoms.append(atom)
            cur_paths.append(sec.path)
            cur_tokens += atom.tokens
        emit()
    emit()
    return chunks


def plan_chunks(doc: Document, target: int, max_tokens: int, hard_max: int, slots: int) -> tuple[list[Chunk], int, int, list[str]]:
    """추출 호출 슬롯 수에 맞도록 묶음 크기를 키워 본다.

    표·문단 의미를 훼손하지 않는 범위(원자 경계)에서만 키우며, hard_max를 넘기지 않는다.
    그래도 슬롯을 넘으면 그대로 반환하고 호출자가 부분 완료로 처리한다.
    """
    notes: list[str] = []
    t, m = target, max_tokens
    chunks = chunk_document(doc, t, m)
    while len(chunks) > slots and m < hard_max:
        t = min(int(t * 1.3), hard_max)
        m = min(max(int(m * 1.3), t), hard_max)
        chunks = chunk_document(doc, t, m)
    if (t, m) != (target, max_tokens):
        notes.append(f"호출 예산에 맞추려고 청크 목표/상한을 {target}/{max_tokens} → {t}/{m} 토큰으로 조정했다.")
    if len(chunks) > slots:
        notes.append(f"청크 {len(chunks)}개가 추출 호출 슬롯 {slots}개를 넘는다: 일부 청크는 처리하지 못하고 부분 완료로 남는다.")
    return chunks, t, m, notes
