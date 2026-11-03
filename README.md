# paper_extractor

`docs/LLM기반_논문정보_추출시스템_개발계획서_v0.1.md`의 구현. PDF → Markdown → 청킹 → 연구 유형 판정 → 청크별 추출 → 통합·검증 → 논문별 JSON → (검수 후) Neo4j.

## 설치

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,neo4j]"
```

`pymupdf4llm`은 `pyproject.toml`에서 버전을 고정한다(변환 결과가 버전에 따라 달라지므로).

## 사용

```bash
# 1) API 호출 없이 변환·청킹·예상 호출 수 확인 (먼저 Markdown/청크를 눈으로 대조)
.venv/bin/python -m paper_extractor plan "논문.pdf" --out outputs

# 2) 추출 (논문당 API 호출 15회 이내)
export OPENAI_API_KEY=...            # 환경변수로만 받는다
export PAPER_EXTRACTOR_MODEL=...     # 모델명은 설정으로 선택 (코드에 고정하지 않음)
.venv/bin/python -m paper_extractor extract "논문.pdf" --out outputs

# 3) outputs/<논문>/graph.json, review.json을 사람이 원문과 대조 → 검토 완료 표시
.venv/bin/python -m paper_extractor mark-reviewed outputs/<논문>/graph.json --reviewer 이름

# 4) 검토 완료본만 Neo4j 적재 (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 필요)
.venv/bin/python -m paper_extractor load-neo4j outputs/<논문>/graph.json [--replace]
```

테스트: `.venv/bin/python -m pytest` (가짜 LLM 사용, API 호출 없음).

## 제목 계층 추출 (LLM 기반, 단계별 개발 중)

논문 제목 계층을 규칙이 아니라 LLM으로 찾는다. 속성값·관계 추출은 이 단계 이후로 보류했고 위 `extract` 파이프라인과는 별개로 동작한다.

```bash
# 0) 키와 모델명은 paper_extractor/.env 에 적는다 (환경변수가 이미 있으면 그 값을 쓴다)
#    OPENAI_API_KEY=...
#    PAPER_EXTRACTOR_MODEL=...

# 1) PDF → Markdown (outputs/markdown/<논문>.md)
.venv/bin/python -m paper_extractor.convert_pdf_md [PDF 파일 또는 폴더]

# 2) 청킹 확인만 (LLM 호출 없음)
.venv/bin/python -m paper_extractor.extract_title --dry-run

# 3) 제목 계층 추출 (outputs/titles/<번호>_title.json, <번호>_title.outline.md)
.venv/bin/python -m paper_extractor.extract_title [md/pdf 파일 또는 폴더]
```

인자를 생략하면 `convert_pdf_md`는 `paper_extractor/papers/`, `extract_title`은 `outputs/markdown/`을 읽는다. `--out`, `--model`, `--max-calls`로 바꿀 수 있다.

### 동작

| 파일 | 하는 일 |
|---|---|
| `convert_pdf_md.py` | pymupdf4llm으로 변환. 2단 편집은 `pdf_to_md.reorder_columns`로 읽기 순서를 바로잡고, 수식·그림으로 오인돼 빠진 짧은 제목(예: `Ⅰ. 서 론`)은 PDF 텍스트 레이어에서 되살려 `##` 후보로 넣는다. 그림 표지(`==> picture ...`)와 `<br>`은 정리한다. |
| `extract_title.py` | 아래 순서로 제목 계층과 섹션별 본문을 만든다. |

1. 띄어쓰기가 사라진 논문은 LLM 호출 없이 제외한다(한글 덩어리 평균 길이 5 초과. 정상 논문은 약 2.6~2.9). 03번이 해당한다.
2. 비어 있지 않은 줄에 `L<번호>|`를 붙여 섹션(`#` 제목 줄로 시작하는 구간) 단위로 청킹한다. 다음 섹션이 예산에 안 들어가면 그 앞에서 자르고, 섹션 하나가 예산보다 크면 문단 경계에서 나눈다.
3. 청크마다 LLM이 제목 줄 번호·레벨·제목만 반환한다. 다음 청크에는 직전 주요 섹션(레벨 2 이하)과 그 이후 제목만 컨텍스트로 넘긴다. `#`와 그 개수는 후보일 뿐이라고 프롬프트에 알린다. 제목의 번호(`1.` 등)는 제거하고 원문 표기는 오탈자도 그대로 둔다.
4. 코드가 제목을 줄 순서로 정렬해 계층 트리를 만들고(2단계 이상 깊어지는 레벨은 보정) 제목 사이 구간을 섹션별 본문으로 묶는다. LLM은 원문을 다시 쓰지 않는다.
5. 점검 결과를 `checks`에 남긴다: 제목 문구가 해당 줄에 없는 경우, LLM이 채택하지 않은 `#` 후보, 추정 제목.

제목이 인쇄되지 않은 서론은 LLM이 `서론`/`Introduction`을 추정 제목(레벨 2, `inferred`)으로 보고할 수 있다. 결과에 `추정 제목`으로 표시되며 본문은 첫 줄부터 시작한다.

### 알려진 한계

- **레벨 판단은 LLM 몫이라 같은 입력도 실행마다 달라질 수 있다.** 서식(글자 크기)은 Markdown에 없어서 LLM이 볼 수 없다. 완료된 호출은 `outputs/cache/`에 저장되어, 같은 입력과 프롬프트는 캐시를 지우지 않는 한 같은 결과가 나온다.
- 제목이 텍스트 레이어에 없는 논문(제목이 도형·이미지)은 복구하지 못한다. 03번이 이 경우라 제외했다.
- `#` 없이 굵은 글씨로만 된 제목은 청킹의 섹션 경계로 인식하지 않는다.
- 파일 이름은 `<논문 번호>_title`이다. 긴 한글 파일명은 OS 길이 제한(255바이트)을 넘어 잘리기 때문이다. 원본 이름은 JSON의 `source`에 있다.
- 실제 LLM 결과를 사람이 확인한 논문은 `papers/`의 6편(03번 제외)이다. 그 밖의 논문 정확도는 확인하지 않았다.

## 산출물 (`outputs/<논문>-<해시>/`)

| 파일 | 내용 |
|---|---|
| `paper.md`, `blocks.json` | 변환 Markdown, 블록 ID·페이지·bbox·제외(참고문헌) 구간, 판독 이상 신호 |
| `chunks.json` | 청크별 제목 경로·구성 블록·토큰 추정·처리 상태·경고 |
| `classification.json` | 1단계 연구 유형 판정 |
| `extractions/` | 청크별 부분 추출 결과(이미지 보완은 `img-p*.json`) |
| `graph.json`, `review.json` | 최종 그래프(`review_status: unreviewed`로 시작), 검토 목록 |
| `usage.json`, `run_manifest.json` | 시도별 실제 토큰 사용량·실패 시도, 설정·버전·청크 상태 |
| `cache/` | 완료된 호출 결과 (PDF·모델·스키마·프롬프트 버전·입력 해시가 키). 재실행 시 호출 없이 재사용 |

## 계획서 요구 → 구현 위치

| 요구 | 구현 |
|---|---|
| 호출 총 15회(재시도 포함, 계획서 10회에서 상향), SDK 숨은 재시도 차단 | `llm.py` `CallBudget`, `OpenAI(max_retries=0)`. 시도 전에 집계 |
| 이미지 보완 최대 5페이지(계획서 2페이지에서 상향), 참고문헌 페이지 제외 | `pipeline._image_supplement`, `PageInfo.is_reference_only` |
| 통합 1회 예약, 예상 호출 표시, 묶음 크기 조절 | `Pipeline.plan`, `chunking.plan_chunks` |
| 첫 페이지 판독 불가 시 판정 전 보완 | `pipeline._transcribe_first_page` |
| 불완전·거절·오류를 성공으로 저장 안 함 | `OpenAIClient.parse` (status/refusal 검사), 캐시는 성공 때만 기록 |
| 근거 ID는 코드가 관리, LLM은 선택 | `pdf_to_md.py` 블록 ID, `pipeline.sanitize_extraction`, `validation.py` |
| 연구 유형별 허용 관계·방향, 종합 집단 PARTICIPATES_IN 금지 | `validation.ALLOWED` |
| 미처리 구간이 있으면 미보고 단정 금지 | `validation`(not_reported → unverifiable) |
| 통합 입력 초과 시 조용히 자르지 않고 보류 | `pipeline._merge` (`partials.json` 저장) |
| 자동 JSON 출력과 DB 쓰기 분리, 검토 전 적재 거부 | `neo4j_loader.py` |

## 구현 기본안 (계획서에서 '미정'인 항목)

계획서 9절의 미정 항목은 확정하지 않았고, 동작에 필요한 잠정 선택만 했다. 바꾸려면 아래 위치만 고치면 된다.

- **JSON 키·자료형**: 모든 속성은 `Attr{value, status, evidence_block_ids, note}`이며 값은 원문 표현 문자열이다(연령 포함, 숫자 정규화 없음). `models.py`.
- **COMPARED_WITH(비교집단)**: 허용 관계에 넣지 않았다. 표현하지 못하는 정보는 `review_items`로 보존한다. `validation.ALLOWED`.
- **Neo4j 매핑**: 속성→`키/키_status/키_evidence`, 활동 구성→`activities_json` 문자열, 논문 ID로 네임스페이스, 재적재는 `--replace`. 버전 보관 없음. `neo4j_loader.py` 상단 주석.
- **모델·출력 한도**: 모델명은 필수 설정, 호출별 출력 한도는 `Config` 기본값(조정 가능).

## 검증 상태 (정직하게)

- 단위·통합 테스트 25개는 가짜 LLM과 합성 PDF로만 돌렸다. **실제 OpenAI 호출, 실제 두 논문 PDF 처리, Neo4j 적재는 실행하지 않았다.**
- 두 논문 PDF(`~/Downloads`)는 이 환경에서 접근이 막혀 변환 품질(다단·표·각주)을 확인하지 못했다. `plan`으로 `paper.md`와 `chunks.json`을 먼저 원본과 대조한다.
- 기계 검증(스키마·ID·방향·숫자 등장 여부)을 통과해도 사람 검수 완료가 아니다. 정확성은 원문에서 만든 비교 기준과 대조해야 한다.
