import numpy as np
import pytest
import torch
from sage_sr.policy import SymbolicPolicy
from sage_sr.priors import StructuredPrior, compile_prior, LLMController


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_film_identity_then_learns_with_demonstrations():
    torch.manual_seed(3)
    policy = SymbolicPolicy(1, hidden_size=16)
    operators = ["+", "R"]
    compiled = compile_prior(StructuredPrior(operator_prior={"+": 0.9, "R": 0.2}), ["x0"], operators)
    h = torch.randn(1, 2, 16)
    assert torch.equal(policy._film(h, compiled.condition), h)
    before = policy.film.weight.detach().clone()
    policy.distill([([0, 2, 1], 0.9)], compiled.condition, operators, epochs=1)
    assert not torch.equal(policy.film.weight, before)


def test_callback_and_teacher_forcing_use_same_legal_distribution():
    policy = SymbolicPolicy(1, hidden_size=16)
    operators = ["+", "R"]
    compiled = compile_prior(None, ["x0"], operators)
    def legal(prefix):
        return [0, 2] if not prefix else [1, 2]
    cb = policy.prior_callback(operators, compiled, legal_actions=legal)
    p = cb([])
    assert p[1] == 0 and sum(p) == pytest.approx(1.0)
    first = policy.log_likelihood([2], operators, compiled.condition, legal)
    assert first.item() == pytest.approx(np.log(p[2]), abs=1e-6)


def test_unspecified_tokens_remain_neutral_and_mutation_weights_positive():
    prior = StructuredPrior(operator_prior={"+": .9, "sin": .1}, structural_family={"rational": .7})
    compiled = compile_prior(prior, ["x0"], ["+", "sin", "R"])
    assert "x0" not in compiled.token_bias and "R" not in compiled.token_bias
    assert compiled.token_bias["+"] > 0 > compiled.token_bias["sin"]
    assert min(compiled.mutation_weights) > 0 and compiled.mutation_weights[3] > 1


def test_llm_json_request_has_no_implicit_retry(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError()
    monkeypatch.setattr("sage_sr.priors.request.urlopen", fail)
    controller = LLMController(base_url="http://localhost:1/v1", model="test-model")
    with pytest.raises(RuntimeError):
        controller.generate({}, [], ["+"])
    assert controller.calls == len(calls) == 1
