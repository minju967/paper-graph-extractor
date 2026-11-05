# paper_extractor

논문 PDF에서 GraphRAG(어떤 질병에 어떤 운동·식단이 도움이 될 수 있는가)용 정보를 뽑는 파이프라인이다.

**목적**: 논문들이 실험한 운동·식단으로 건강 증진 프로그램을 구성해, 보험사 담당자의 프로그램 설계나 건강앱 사용자의 프로그램 선택을 돕는다.
"이 프로그램을 하면 건강이 좋아진다"가 아니라 "이런 프로그램이 도움이 될 수 있다" 정도의 추천이므로 DB의 수치가 정확할 필요는 없고,
**질병 개선에 도움이 되는 프로그램과 그 수행 결과를 잘 찾을 수 있어야 한다.**

현재 구현 범위는 아래와 같다.

```
PDF → Markdown(+메타데이터) → 제목 계층(LLM) → 섹션 청크 → 라우팅(LLM 1회) → Paper·Population·Program 노드 속성(노드마다 LLM 1회) → Outcome 속성(Program마다 LLM 1회)
```

노드 사이의 관계 생성, 검색, 속성 값 검증은 아직 없다. 설계와 확정 사항은 `docs/프로젝트_기획안.md`,
추출 기준은 `docs/GraphRAG_추출기준_v0.5.md`에 있다(이 저장소 밖 `personal_AI/docs/`).

## 설치와 설정

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

키와 모델명은 `paper_extractor/.env`(git 제외)에 적는다. 환경변수가 이미 있으면 그 값을 쓴다.

```
OPENAI_API_KEY=...
PAPER_EXTRACTOR_MODEL=...
```

`pymupdf4llm`은 변환 결과가 버전에 따라 달라지므로 `pyproject.toml`에서 버전을 고정했다.
테스트: `.venv/bin/python -m pytest` (LLM 호출 없음).

## 사용

LLM을 호출하는 단계는 표에 표시했다. LLM은 논문 한 편의 한 단계에 호출 횟수 한도(`CallBudget`)가 있고, 같은 입력은
`outputs/cache/`의 캐시를 써서 다시 호출하지 않는다.

| 단계 | 명령 | LLM |
|---|---|---|
| 1. 변환·제목·청크 일괄 | `.venv/bin/python -m paper_extractor.build_all [PDF 파일 또는 폴더]` | 제목이 바뀐 논문만 |
| 2. 라우팅(노드별 섹션 선택) | `.venv/bin/python -m paper_extractor.routing_call outputs/chunks/NN_chunks.json` | 1회 |
| 3. Paper 입력 확인 | `.venv/bin/python -m paper_extractor.paper.inputs outputs/chunks/NN_chunks.json [--full]` | 없음 |
| 4. Paper 속성 추출 | `.venv/bin/python -m paper_extractor.paper.call outputs/chunks/NN_chunks.json` | 1회 |
| 5. Population 속성 추출 | `.venv/bin/python -m paper_extractor.population.call outputs/chunks/NN_chunks.json --study-type "중재 연구"` | 1회 |
| 6. Program 속성 추출 | `.venv/bin/python -m paper_extractor.program.call outputs/chunks/NN_chunks.json --study-type "중재 연구" [--evidence] [--cache-only]` | 1회 |
| 7. Outcome 입력 확인 | `.venv/bin/python -m paper_extractor.outcome.inputs outputs/chunks/NN_chunks.json --study-type "중재 연구" [--full]` | 없음 |
| 8. Outcome 속성 추출 | `.venv/bin/python -m paper_extractor.outcome.call outputs/chunks/NN_chunks.json --study-type "중재 연구" --split [--evidence] [--cache-only]` | 대조가 아닌 Program 수만큼 |

인자를 생략하면 PDF는 `paper_extractor/papers/`를 읽는다. 3·4단계는 2단계의 캐시된 라우팅 결과를 쓰며, 캐시가 없으면
3단계는 호출하지 않고 멈춘다. Outcome(7·8단계)은 같은 논문의 Population·Program 결과를 캐시에서만 읽으므로 5·6단계를 먼저 실행해야
하고, 캐시에 없으면 호출하지 않고 멈춘다. LLM 호출이 시간 초과 등으로 실패하면 제한 시간을 1.5배씩 늘려 최대 2회 더 시도한다(`config.py`). 더 작은 단위의 명령도 있다.

```bash
.venv/bin/python -m paper_extractor.convert_pdf_md [PDF 파일 또는 폴더]          # PDF → Markdown, 메타데이터
.venv/bin/python -m paper_extractor.extract_title [--dry-run] [md/pdf 파일 또는 폴더]  # 제목 계층 (LLM)
.venv/bin/python -m paper_extractor.chunk_sections [*_title.json 파일 또는 폴더]    # 섹션 청크 (LLM 없음)
.venv/bin/python -m paper_extractor.routing_input outputs/chunks/NN_chunks.json    # 라우팅 입력 보기 (LLM 없음)
.venv/bin/python -m paper_extractor.evidence_refs outputs/chunks/NN_chunks.json 1  # 문장 번호 붙인 청크 보기 (LLM 없음)
```

## 코드 구성

공통 코드는 `paper_extractor/`, 노드별 속성 추출 코드는 노드 이름의 하위 폴더(`paper/`, `population/`, `program/`, `outcome/`)에 둔다. 노드 폴더는 `prompt.py`, `inputs.py`, `call.py`로 이루어진다.

| 파일 | 하는 일 |
|---|---|
| `convert_pdf_md.py` | pymupdf4llm으로 PDF를 Markdown으로 변환. 머리글·바닥글은 본문에서 빼 `<번호>_meta.json`에 보관, 2단 편집 순서 보정, 깨진 부호(`�`) 복원, 표·그림 캡션 정리 |
| `extract_title.py` | LLM으로 제목 계층을 뽑고 섹션별 본문으로 묶는다. 이전에 확정한 구조를 다시 붙이는 `reanchor`(LLM 없음) 포함 |
| `chunk_sections.py`, `figures.py` | 섹션을 토큰 예산(기본 4000)에 맞춰 청크로 묶는다. 표는 통째로, 깨진 표·그림은 PNG로 잘라 붙인다. 참고문헌 제외 |
| `build_all.py` | 변환 → 제목 → 청크를 한 번에 돌리고 달라진 논문만 다시 처리 |
| `routing_input.py`, `routing_prompt.py`, `routing_call.py` | 섹션 목록(ID·제목 경로·첫 문단 앞부분)을 LLM에 보내 노드마다 1순위·후보 섹션을 받는다. 캐시 전용 읽기(`cached_routing`) 포함 |
| `evidence_refs.py` | 청크의 섹션 본문을 문장으로 나누고 `섹션ID.s번호`를 붙인다. LLM은 근거를 번호로만 답하고 코드가 원문 문장과 섹션 경로로 바꾼다 |
| `attrs.py` | 노드 속성의 공통 형식 `{value, evidence, note}`, 코드가 값 유무로 채우는 정보 상태 `status`(보고됨 / 미보고), 연구 유형 목록 |
| `paper/prompt.py` | Paper 시스템 프롬프트와 출력 스키마(속성 11개, 각 `{value, evidence, note}`) |
| `paper/inputs.py` | 라우팅 결과로 보낼 청크 선택, 메타데이터 줄과 DOI 후보 합치기, 입력 조립 |
| `paper/call.py` | LLM 호출, 단일 DOI 후보의 코드 채택, 근거 번호 → 원문 교체와 출력 |
| `population/` | 집단 목록 추출(질병은 4개 분류로 정규화, 연령은 정수). 연구 유형을 첫 줄에 알려 주고, 중재·관찰 연구는 `population` 1순위+후보 1순위 청크, 문헌 종합 연구는 `population`+`outcome` 1순위 청크를 보낸다 |
| `program/` | 프로그램(조건) 목록과 활동 구성 추출. `program` 1순위 청크를 보낸다. 대조 조건도 Program, 준비·정리운동은 note, 문헌 종합 연구는 운동 종류×프로그램 출처로 나눈다 |
| `outcome/` | 결과 목록 추출. `outcome` 1순위 청크와 앞서 추출한 Program(P1…)·Population(G1…) 목록을 보내고, LLM이 결과마다 연결 번호를 답한다. `--split`이면 대조가 아닌 Program마다 1회씩 호출해 합치며(`direction.py`는 신뢰구간 판정, `call.py`는 중복 제거) 출력은 모든 속성을 값이 없는 것까지 보여 준다 |
| `llm.py`, `config.py` | OpenAI 호출·호출 한도·캐시·시간 초과 재시도, 설정 |
| `pdf_to_md.py`, `chunking.py`, `models.py` | 옛 파이프라인에서 남은 보조 코드. 변환·청킹이 일부 함수만 쓴다 |

### Paper 속성

`title`, `authors`, `year`, `journal`, `study_type`(필수), `review_subtype`, `doi`, `inclusion_criteria`,
`exclusion_criteria`, `recruit_age_min`, `recruit_age_max`(선택). 값이 없으면 `null`이다.

- 근거는 LLM이 인용문을 쓰지 않고 입력의 문장 번호만 답한다(`title`은 근거 없음). 입력에 없는 번호는 오류로 표시한다.
- 입력은 라우팅이 고른 1순위 섹션과 후보 1순위 섹션이 있는 청크 전체, 그리고 메타데이터(머리글·바닥글 줄)다.
- DOI 후보가 하나면 코드가 채택하고, 둘 이상이면 후보를 프롬프트에 붙여 LLM이 고르며, 없으면 비워 둔다.
- 모집 연령은 숫자 최소·최대로 적고 원문 표현은 note에 남긴다(`40~50대` → 40/59).
- 연구 유형: 중재 연구 / 관찰 연구 / 문헌 종합 연구. 연구자가 중재를 배정했는지 애매하면 중재 연구로 본다.
- 닫힌 값(활동 유형·프로그램 출처 등) 점검은 LLM이 아니라 코드가 맡는다. Population과 Outcome은 구현했고 Paper·Program은 아직 없다.
- `Population`, `Program`의 연구 유형은 아직 `Paper` 결과와 연결하지 않고 `--study-type`으로 받는다.

### 정보 상태

모든 속성에 `status`가 있다. 논문에서 값을 찾았으면 `보고됨`, 없으면 `미보고`이며, **LLM이 답하지 않고 코드가 값 유무로 정한다.**
출력에는 `속성: '값'  [보고됨]`처럼 표시한다. OpenAI에 보내는 스키마에는 들어가지 않는다.

### Population 속성

`condition`(필수), `age`, `sex`, `risk_group`, `group_name`, `group_level`, `group_role`.

- `condition`은 **혈압 / 혈당 / 간수치 / 비만 / 기타** 중 하나다. 원문 표현은 note에 적는다. 질병이 여럿이면 연구가 주로 겨냥하는 하나를 고르고 전부를 note에 적는다.
  `기타`는 네 분류에 없는 질병이나 건강한 성인이다.
- `age`는 **정수**다. 평균 나이에서 표준편차를 버리고 소수점을 반올림 없이 자른다(`46.08±4.51세` → 46). 원문은 note에 적는다. 평균이 없으면 null이다.
- 집단 역할은 중재군 / 대조군 / 전체 분석 집단 / 하위집단이다. 집단이 하나면 `전체 분석 집단`이다. 문헌 종합 연구는 논문이 결과나 권고를 직접 제시하는 주제 집단만 만들고,
  전체 대상과 범주별(예: 혈압 수준별) 결과를 모두 보고하면 둘 다 만든다.

### Outcome 속성

결과 하나는 지표 × Program × 집단 × 비교 대상 × 비교 방식이다. 속성은 `indicator`, `indicator_category`, `direction`, `change_direction`,
`significance`, `significance_statistics`, `representative_values`, `comparison_type`, `measurement_time`(각 `{value, evidence, note}`)과
연결 `program_ids`, `population_ids`, `comparison_program_ids`, `link_evidence`, `link_note`이다.

- 닫힌 값: `indicator_category` = 혈압 / 혈당 / 간수치 / 비만 / 기타, `direction` = 감소 / 동일 / 증가, `change_direction` = 감소 / 증가,
  `significance` = 유의 / 비유의, `comparison_type` = 집단 내 전후 / 통합 분석 / 조건 간 / 집단 간·상호작용. 목록을 어기면 출력에 오류로 표시한다.
- **`direction`은 LLM이 아니라 코드가 정한다.** LLM은 변화 방향(`change_direction`)과 유의성(`significance`)만 답한다. 유의성이 `비유의`이면 `동일`, `유의`이면 변화 방향을 쓴다.
  유의성이 비어 있고 신뢰구간이 하나 있으면 구간으로 판정한다(전체 음수 감소, 전체 양수 증가, 0을 지나면 동일). 유의성을 말하지 않고 변화 방향만 있으면 `direction`은 null이다.
  신뢰구간·p값(0.05 기준)이 LLM이 답한 유의성·방향과 어긋나면 값을 바꾸지 않고 출력에 `!!`로 표시한다.
- `동일`은 논문이 "유의한 차이 없음"이라고 한 경우다. 프로그램끼리 직접 비교한 결과는 문장의 주어(더 낫다고 보고된 쪽)가 평가 Program이고, 대칭 비교는 번호가 작은 Program이 평가 Program이다.
- 비교 방식 판단 순서: 한 조건·한 집단의 전후 → `집단 내 전후` / 문헌 종합 연구의 Program 하나 대 대조(하위집단별 포함) → `통합 분석` / 두 Program 직접 비교 → `조건 간` / 서로 다른 조건을 받은 집단끼리 → `집단 간·상호작용`.
- 하위집단끼리만 비교한 결과는 Outcome으로 만들지 않고 각 하위집단의 효과를 따로 만든다. 본문과 수치가 반대이면 본문 값을 쓰고 note에 적는다. 지표 표기는 원문 그대로 둔다(`DDBP`도 그대로).
- 대조군과 비교한 합산 효과 등 통계는 그 결과 자체의 검정값만 쓴다. 하위집단 간 차이 검정의 p값을 각 하위집단 결과에 복사하지 않는다. `note`는 한국어로 쓴다.
- 연결 번호는 입력 목록에 있는 것만 인정하고, Program은 `P`, 집단은 `G`로 시작해야 한다. 호출은 대조가 아닌 Program마다 1회이고, 합칠 때 같은 비교 쌍의 중복을 코드로 제거한다.

### 검색 용도 (설계, 아직 구현 안 함)

입력은 질병, 성별, 연령(40대, 50대 같은 범위)이고 결과는 추천할 프로그램이다. 대상(Population)에서 시작해 Outcome을 거쳐 Program을 찾는다.

- Program–Population을 따로 저장하지 않는다. Outcome의 `population_ids`와 `program_ids`가 이어 준다.
- 성별이나 연령이 `미보고`인 Population도 후보에 넣고 순위를 낮춘다. Population의 `age`(정수)가 입력 연령 범위 안에 들어오면 후보로 뽑는다.
- 질병은 Population의 `condition`과 Outcome의 `indicator_category`로 맞추고, 개선 결과(`direction`이 `감소`)를 고른다. 순위는 비교 방식으로 가늠한다.
  이 경로와 후보 조건은 논의 중이며 기획안 v1.41에 `[검토 중]`으로 있다.

## 산출물 (`outputs/`, git 제외)

| 위치 | 내용 |
|---|---|
| `markdown/<논문>.md`, `<번호>_meta.json` | 변환 Markdown, 머리글·바닥글·DOI 후보 등 메타데이터 |
| `titles/<번호>_title.json`, `.outline.md` | 제목 계층과 섹션별 본문 |
| `chunks/<번호>_chunks.json`, `.md`, `<번호>_figures/` | 청크(섹션 ID·경로 포함)와 잘라 낸 그림·표 이미지 |
| `cache/` | 완료된 LLM 호출 결과. 입력·프롬프트가 같으면 재사용 |

파일 이름은 긴 한글 파일명이 OS 길이 제한을 넘지 않도록 `<논문 번호>`를 쓴다. 원본 이름은 JSON의 `source`에 있다.

## 알려진 한계

- 제목 레벨 판단과 속성 추출은 LLM이라 입력이나 프롬프트가 바뀌면 결과가 달라질 수 있다. 같은 입력·프롬프트는 캐시에서 같은 결과가 나온다. 같은 프롬프트를 여러 번 새로 호출했을 때 얼마나 달라지는지는 측정하지 않았다.
- 띄어쓰기가 사라진 논문(한글 덩어리 평균 길이 5 초과)은 제목 추출에서 제외한다(03번). 제목이 텍스트 레이어에 없는 논문은 복구하지 못한다.
- PDF 변환에서 생긴 띄어쓰기 오류(`최 근`, `30 대`)는 속성값에도 그대로 남는다(원문 표현 유지 규칙).
- 문장 분리는 규칙 기반이라 약어·표에서 경계가 어긋날 수 있다.
- 실제 LLM 결과를 확인한 논문은 `papers/`의 6편(01, 02, 04, 05, 06, 07번)이다. 그 밖의 논문 정확도는 확인하지 않았다.
- Outcome 시험 결과: 07번 26개, 01번 12개, 02번 14개, 04번 8개, 05번 16개, 06번 7개. `direction`은 83개 중 59개에서 정해지고 나머지는 null이다(집단×시기 상호작용, 교차 설계의 조건 간,
  유의성을 말하지 않은 서술형 리뷰). 06번(서술형 리뷰)은 Program 6개 중 4개에서 Outcome이 없어 누락 가능성이 있으나 확인하지 않았다.
- 07번 Population 재호출에서 전체 대상 집단이 빠진 적이 있다(프롬프트 보강 후 재확인 전). 질병이 여럿인 집단은 하나로 줄어 note에만 남는다. 연령은 평균 나이 하나뿐이다.
- 표본 수와 연구 설계 같은 근거의 강도는 속성으로 두지 않았다. 문헌 종합 연구는 성별·연령이 없다.
- 상세한 결정 이력과 미결 사항은 `docs/프로젝트_기획안.md`의 변경 이력(v1.29~v1.42)에 있다.
