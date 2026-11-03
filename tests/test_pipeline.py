import json

import pytest

from paper_extractor.llm import BudgetedLLM, BudgetExceeded, CallBudget, LLMError
from paper_extractor.models import TypeDecision
from paper_extractor.neo4j_loader import build_rows, load_graph, load_into_neo4j, mark_reviewed
from paper_extractor.pipeline import Pipeline

from conftest import FakeClient


def test_dry_run_makes_plan_without_client(sample_pdf, cfg):
    res = Pipeline(cfg, None, log=lambda *_: None).run(sample_pdf, dry_run=True)
    assert res.graph is None
    assert (res.workdir / "chunks.json").exists() and (res.workdir / "paper.md").exists()


def test_full_run_with_fake_llm_writes_outputs_and_respects_budget(sample_pdf, cfg):
    client = FakeClient()
    res = Pipeline(cfg, client, log=lambda *_: None).run(sample_pdf)
    assert res.graph is not None and res.graph.status.review_status == "unreviewed"
    assert len(client.calls) <= cfg.max_api_calls
    assert client.calls[0] == "classify" and client.calls[-1] == "merge"
    for name in ("graph.json", "review.json", "usage.json", "classification.json", "run_manifest.json", "blocks.json"):
        assert (res.workdir / name).exists(), name
    usage = json.loads((res.workdir / "usage.json").read_text("utf-8"))
    assert usage["totals"]["api_calls_attempted"] == len(client.calls)


def test_rerun_uses_cache_and_makes_no_calls(sample_pdf, cfg):
    Pipeline(cfg, FakeClient(), log=lambda *_: None).run(sample_pdf)
    second = FakeClient()
    res = Pipeline(cfg, second, log=lambda *_: None).run(sample_pdf)
    assert second.calls == [] and res.graph is not None


def test_failed_chunk_is_not_success_and_graph_marked_partial(sample_pdf, cfg):
    client = FakeClient(fail_stages={"extract-c01"})
    res = Pipeline(cfg, client, log=lambda *_: None).run(sample_pdf)
    assert res.graph is not None and not res.graph.status.complete
    assert any("c01" in r for r in res.partial_reasons)


def test_classification_failure_aborts_without_graph(sample_pdf, cfg):
    res = Pipeline(cfg, FakeClient(fail_stages={"classify"}), log=lambda *_: None).run(sample_pdf)
    assert res.graph is None and "중단" in res.message


def test_low_text_page_triggers_image_supplement_at_most_two_pages(sample_pdf, cfg):
    cfg.min_chars_per_page = 100000  # 모든 페이지가 판독 이상으로 표시됨
    client = FakeClient()
    res = Pipeline(cfg, client, log=lambda *_: None).run(sample_pdf)
    images = [c for c in client.calls if c.startswith(("image", "transcribe"))]
    assert 1 <= len(images) <= 2 and len(client.calls) <= 10
    assert res.graph is not None


def test_budget_never_exceeded_even_with_retries():
    class Flaky:
        def parse(self, **kw):
            raise LLMError("timeout", retryable=True)

    budget = CallBudget(3)
    llm = BudgetedLLM(Flaky(), budget, __import__("pathlib").Path("/tmp/pe_test_cache_x"), {}, "m")
    with pytest.raises(LLMError):
        llm.run(stage="s", system="a", user=str(id(budget)), schema=TypeDecision, max_output_tokens=10, retries=10)
    assert budget.used == 3
    with pytest.raises(BudgetExceeded):
        budget.consume()


def test_loader_requires_review_and_builds_flat_rows(sample_pdf, cfg):
    res = Pipeline(cfg, FakeClient(), log=lambda *_: None).run(sample_pdf)
    gpath = res.workdir / "graph.json"
    graph = load_graph(gpath)
    with pytest.raises(PermissionError):
        load_into_neo4j(graph)
    mark_reviewed(gpath, "tester")
    graph = load_graph(gpath)
    assert graph.status.review_status == "reviewed"
    nodes, rels = build_rows(graph)
    prog = next(p for lbl, p in nodes if lbl == "Program")
    assert prog["uid"].endswith(":prg_1") and isinstance(json.loads(prog["activities_json"]), list)
    assert all(isinstance(v, (str, list)) for _, p in nodes for v in p.values())
    assert {r["type"] for r in rels} == {"HAS_POPULATION", "PARTICIPATES_IN", "HAS_OUTCOME", "MEASURED_IN"}
