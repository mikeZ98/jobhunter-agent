import pytest
from pydantic import SecretStr

from jobhunter.llm.router import ModelRouter, NoModelAvailable, Provider, Task, decision_matrix

CODER, LLAMA, BIG = "qwen2.5-coder:7b", "llama3.1:8b", "qwen2.5:14b"


@pytest.fixture
def router(settings, monkeypatch):
    settings.llm.fast_models = [CODER, LLAMA]
    settings.llm.creative_models = [BIG, LLAMA]
    settings.llm.interview_models = [BIG, LLAMA]
    r = ModelRouter(settings)
    monkeypatch.setattr(r.probe, "refresh", lambda force=False: None)
    r.probe._ollama_up = True
    r.probe._installed = {f"{CODER}", f"{LLAMA}", f"{BIG}"}
    r.probe._resident = set()
    monkeypatch.setattr(type(r.probe), "free_memory_gb", staticmethod(lambda: 32.0))
    return r


def test_tasks_route_to_configured_local_tags(router):
    assert router.route(Task.SCORE, "short prompt").model.name == CODER
    assert router.route(Task.CREATIVE, "short prompt").model.name == BIG


def test_long_context_skips_32k_models(router):
    router.settings.llm.ollama_num_ctx = 131_072
    d = router.route(Task.EXTRACT, "x" * 200_000, 1000)  # ~57k tokens
    assert d.model.name == LLAMA
    assert any("context" in reason for _, reason in d.skipped)


def test_low_memory_falls_back_to_smaller_local_then_cloud(router, monkeypatch):
    monkeypatch.setattr(type(router.probe), "free_memory_gb", staticmethod(lambda: 10.0))
    assert router.route(Task.INTERVIEW, "prompt").model.name == LLAMA  # 14b (10.5 GB) doesn't fit
    monkeypatch.setattr(type(router.probe), "free_memory_gb", staticmethod(lambda: 6.0))
    router.settings.anthropic_api_key = SecretStr("sk-test")
    d = router.route(Task.INTERVIEW, "prompt")
    assert d.model.provider == "anthropic"
    assert all("GB" in r or "not installed" in r for _, r in d.skipped)


def test_local_only_low_memory_still_tries_smallest_local(router, monkeypatch):
    router.settings.llm.local_only = True
    router.matrix = decision_matrix(router.settings.llm)
    monkeypatch.setattr(type(router.probe), "free_memory_gb", staticmethod(lambda: 4.0))
    c = router.candidates(Task.INTERVIEW, "prompt", 500)
    assert [d.model.name for d in c] == [LLAMA, BIG]  # nothing fits -> smallest first, never cloud
    assert c[0].skipped[-1][1] == "memory tight — may swap"


def test_resident_model_ignores_memory_gate(router, monkeypatch):
    monkeypatch.setattr(type(router.probe), "free_memory_gb", staticmethod(lambda: 1.0))
    router.probe._resident = {f"{BIG}"}  # already loaded in Ollama -> no extra memory needed
    assert router.route(Task.CREATIVE, "prompt").model.name == BIG


def test_nothing_available_raises_with_reasons(router):
    router.probe._ollama_up = False
    router.settings.anthropic_api_key = None
    router.settings.openai_api_key = None
    with pytest.raises(NoModelAvailable, match="ollama not reachable"):
        router.route(Task.SCORE, "prompt")


def test_exclusion_moves_to_next_candidate(router):
    c = router.candidates(Task.SCORE, "prompt", 500, exclude={f"ollama:{CODER}"})
    assert c[0].model.name == LLAMA


def test_local_only_drops_cloud_even_with_keys(settings):
    settings.llm.local_only = True
    settings.anthropic_api_key = SecretStr("sk-test")
    m = decision_matrix(settings.llm)
    assert all(spec.provider == Provider.OLLAMA for specs in m.values() for spec in specs)
    assert [s.name for s in m[Task.SCORE]] == settings.llm.fast_models


def test_unknown_tag_gets_context_and_memory_from_ollama_metadata(router):
    router.settings.llm.fast_models = ["phi4-mini:latest"]
    router.matrix = decision_matrix(router.settings.llm)
    router.probe._installed.add("phi4-mini:latest")
    router.probe._meta = {"phi4-mini": (131_072, 2.5)}
    router.settings.llm.ollama_num_ctx = 131_072
    m = router.route(Task.SCORE, "x" * 200_000, 1000).model
    assert (m.name, m.context_window, m.mem_gb) == ("phi4-mini", 131_072, 4.0)


def test_ollama_context_is_capped_to_num_ctx(router):
    router.settings.llm.ollama_num_ctx = 12_288
    router.settings.anthropic_api_key = router.settings.openai_api_key = None  # shell may export real keys
    assert router.route(Task.EXTRACT, "x" * 30_000, 1000).model.context_window == 12_288  # ~9.6k tokens fits
    with pytest.raises(NoModelAvailable, match="context"):
        router.route(Task.EXTRACT, "x" * 60_000, 1000)  # ~18k tokens: would be truncated -> refuse


def test_ollama_alias_names():
    from jobhunter.llm.router import ollama_alias

    assert ollama_alias("gemma3:4b", 12_288) == "gemma3:4b-jh12k"
    assert ollama_alias("phi4-mini", 16_384) == "phi4-mini:latest-jh16k"


def test_concurrent_refresh_never_reports_ollama_down(settings, monkeypatch):
    """Regression: parallel LLM calls read `_ollama_up=False` while the first probe was in flight."""
    import threading
    import time

    from jobhunter.llm.router import ResourceProbe

    probe = ResourceProbe(settings.llm)

    def slow_probe() -> bool:
        time.sleep(0.2)
        probe._ollama_up = True
        return True

    monkeypatch.setattr(probe, "_probe_ollama", slow_probe)
    seen: list[bool] = []

    def worker() -> None:
        probe.refresh()
        seen.append(probe._ollama_up)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [True] * 8


def test_unloading_the_resident_model_counts_as_free_memory(router, monkeypatch):
    # 16 GB Mac, gemma-sized model resident: 4.5 GB free + 5 GB the client will unload -> 14b still too
    # big, but llama (6.5 GB + 2.5 headroom) fits instead of being "memory tight".
    router.settings.llm.local_only = True
    router.matrix = decision_matrix(router.settings.llm)
    monkeypatch.setattr(type(router.probe), "free_memory_gb", staticmethod(lambda: 4.5))
    router.probe._resident = {"gemma3:4b-jh12k"}
    router.probe._resident_gb = {"gemma3:4b-jh12k": 5.0, "nomic-embed-text:latest": 0.6}
    c = router.candidates(Task.INTERVIEW, "prompt", 500)
    assert c[0].model.name == LLAMA and c[0].skipped[-1][1].startswith("needs 10.5 GB")
    router.settings.llm.ollama_single_model = False  # nothing gets unloaded -> llama doesn't fit either
    assert router.candidates(Task.INTERVIEW, "prompt", 500)[0].skipped[-1][1] == "memory tight — may swap"
