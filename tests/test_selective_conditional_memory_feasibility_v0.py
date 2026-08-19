from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src import selective_conditional_memory_feasibility_v0 as module


ROOT = Path(__file__).resolve().parents[1]


def test_kappa_endpoints_are_frozen_baselines() -> None:
    rng=np.random.default_rng(1); current=rng.normal(size=5); gamma=rng.normal(size=(4,5)); residual=rng.normal(size=(4,5))
    np.testing.assert_allclose(module.prototypes_for_kappa(current,gamma,residual,0), current+gamma)
    np.testing.assert_allclose(module.prototypes_for_kappa(current,gamma,residual,1), current+gamma+residual)


def test_kappa_is_convex_shrinkage() -> None:
    rng=np.random.default_rng(2); current=rng.normal(size=5); gamma=rng.normal(size=(3,5)); residual=rng.normal(size=(3,5)); k=.23
    p0=module.prototypes_for_kappa(current,gamma,residual,0); p1=module.prototypes_for_kappa(current,gamma,residual,1)
    np.testing.assert_allclose(module.prototypes_for_kappa(current,gamma,residual,k),(1-k)*p0+k*p1)


def test_classwise_kappa_shape() -> None:
    result=module.prototypes_for_kappa(np.zeros(3),np.zeros((4,3)),np.ones((4,3)),np.asarray([0,.25,.5,1]))
    np.testing.assert_allclose(result[:,0],[0,.25,.5,1])


def test_invalid_kappa_shape_fails() -> None:
    with pytest.raises(module.NumericalContractError):
        module.prototypes_for_kappa(np.zeros(3),np.zeros((4,3)),np.ones((4,3)),np.ones(3))


def test_quadratic_distance_coefficients() -> None:
    rng=np.random.default_rng(3); trials=rng.normal(size=(12,6)); base=rng.normal(size=(4,6)); residual=rng.normal(size=(4,6)); k=.41
    a,b,c=module._distance_coefficients(trials,base,residual)
    direct=np.sum((trials[:,None,:]-(base+k*residual)[None,:,:])**2,axis=2)
    np.testing.assert_allclose(a+k*b+k*k*c,direct,atol=2e-12,rtol=2e-12)


def test_subject_loss_derivative_matches_finite_difference() -> None:
    rng=np.random.default_rng(4); trials=rng.normal(size=(30,5)); base=rng.normal(size=(3,5)); residual=rng.normal(size=(3,5)); labels=np.repeat(np.arange(3),10); a,b,c=module._distance_coefficients(trials,base,residual)
    example=module.GateExample(0,1,np.zeros(24),np.zeros((3,8)),residual,a,b,c,labels)
    k=.3; loss,derivative=module._subject_loss_and_derivative(example,k); step=1e-6
    finite=(module._subject_loss_and_derivative(example,k+step)[0]-module._subject_loss_and_derivative(example,k-step)[0])/(2*step)
    assert loss>0 and derivative==pytest.approx(finite,rel=1e-6,abs=1e-6)


def _synthetic_examples(seed: int=5) -> tuple[list[module.GateExample],np.ndarray]:
    rng=np.random.default_rng(seed); examples=[]; truth=[]
    for index in range(40):
        h=rng.normal(size=24); k=float(1/(1+np.exp(-(2*h[0]-h[1])))); truth.append(k); classes,dim=3,5; base=rng.normal(size=(classes,dim)); residual=rng.normal(size=(classes,dim)); labels=np.repeat(np.arange(classes),15); trials=np.concatenate([base[c]+k*residual[c]+.2*rng.normal(size=(15,dim)) for c in range(classes)]); a,b,c=module._distance_coefficients(trials,base,residual); examples.append(module.GateExample(index,index+1,h,rng.normal(size=(classes,8)),residual,a,b,c,labels))
    return examples,np.asarray(truth)


def test_gate_learns_synthetic_reliability() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); examples,truth=_synthetic_examples(); model=module.fit_gate(examples,1e-3,config); predicted=np.asarray([module.predict_kappa(model,e.global_features) for e in examples])
    assert np.corrcoef(predicted,truth)[0,1]>.85


def test_gate_deterministic() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); examples,_=_synthetic_examples(); a=module.fit_gate(examples,1e-2,config); b=module.fit_gate(examples,1e-2,config)
    np.testing.assert_array_equal(a.parameters,b.parameters)


def test_vectorized_null_fit_matches_scalar_fit() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); examples,_=_synthetic_examples()
    scalar=module.fit_gate(examples,1e-2,config); vectorized=module.fit_gate_vectorized(examples,1e-2,config)
    np.testing.assert_allclose(vectorized.parameters,scalar.parameters,atol=2e-9,rtol=2e-9)
    assert vectorized.objective==pytest.approx(scalar.objective,abs=2e-12,rel=2e-12)


def test_standardizer_constant_column_scale_one() -> None:
    values=np.column_stack([np.arange(5),np.ones(5)]); mean,scale=module._standardizer(values)
    assert mean[1]==1 and scale[1]==1


def test_reliability_ratio_bounds_synthetic() -> None:
    rng=np.random.default_rng(6); a=rng.normal(size=(4,8)); b=a+.1*rng.normal(size=(4,8)); mean=.5*(a+b); noise=.5*(a-b); rel=np.maximum(np.sum(mean**2,axis=1)-np.sum(noise**2,axis=1),0)/(np.sum(mean**2,axis=1)+1e-12)
    assert np.all(rel>=0) and np.all(rel<=1+1e-12)


def test_fixed_point_free_memory_and_feature_nulls() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False)
    for namespace in ("memory","feature"):
        order=module.parent.fixed_point_free(module._rng(config,namespace),11); assert not np.any(order==np.arange(11))


def test_class_semantics_null_nonidentity() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); order=module.parent.nonidentity_class_permutation(module._rng(config,"class"),4)
    assert not np.array_equal(order,np.arange(4))


def test_oracle_is_at_least_each_endpoint() -> None:
    identity=np.asarray([.4,.7,.6]); population=np.asarray([.6,.5,.6]); oracle=np.maximum(identity,population)
    assert np.all(oracle>=identity) and np.all(oracle>=population)


def test_bootstrap_and_permutation_deterministic() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); values=np.asarray([.1,-.1,.3,.2])
    assert module._bootstrap_ci(values,config,"unit",100)==module._bootstrap_ci(values,config,"unit",100)
    assert module._paired_signflip_p(values,config,"unitp",99)==module._paired_signflip_p(values,config,"unitp",99)


def test_holm_is_monotone() -> None:
    rows=[{"p_value_raw":.04},{"p_value_raw":.01},{"p_value_raw":.03}]; module._holm(rows); ordered=sorted(rows,key=lambda row:row["p_value_raw"])
    assert all(ordered[i]["p_value_holm"]<=ordered[i+1]["p_value_holm"] for i in range(2))


def test_gate_input_api_has_no_deployment_feature_or_label() -> None:
    parameters=inspect.signature(module._reliability_record).parameters
    assert "deployment_features" not in parameters and "deployment_labels" not in parameters


def test_parent_terminal_frozen() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False)
    assert config["protocol"]["parent_terminal"]=="STOP_NO_CROSS_SESSION_DOWNSTREAM_UTILITY"


def test_decision_labels() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False)
    assert config["decisions"]["go"]=="GO_SELECTIVE_MEMORY_FOR_SPDNET"
    assert config["decisions"]["gate_stop"]=="STOP_RELIABILITY_GATE_CANNOT_SELECT_MEMORY"
    assert config["decisions"]["oracle_stop"]=="STOP_NO_SELECTIVE_MEMORY_ORACLE_HEADROOM"


def test_parent_manifest_hashes() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False); observed=module.validate_parent_contract(ROOT,config)
    assert all(f"pr{number}" in observed for number in range(16,21))


@pytest.mark.skipif(not (ROOT/"cache/stieger2021_multiclass_confirmation_v0").exists(),reason="ignored parent cache unavailable")
def test_live_cache_schema_and_acquisition_order_without_statistics() -> None:
    config,_=module.load_config(ROOT,verify_protocol=False)
    for dataset,subjects in (("stieger",62),("openbmi",54)):
        bundle=module.load_dataset(ROOT,config,dataset); order=module.acquisition_orders(bundle)
        assert len(order)==subjects and all(len(np.unique(value))==len(value) for value in order.values())


def test_result_manifest_excludes_itself(tmp_path: Path) -> None:
    (tmp_path/"x.txt").write_text("x"); (tmp_path/"manifest.json").write_text("old")
    manifest=module._result_manifest(tmp_path)
    assert [row["path"] for row in manifest["records"]]==["x.txt"]


def test_reverse_diagnostic_is_non_voting_in_protocol() -> None:
    protocol=(ROOT/"docs/PROTOCOL_SELECTIVE_CONDITIONAL_MEMORY_FEASIBILITY_V0.md").read_text()
    assert "Reverse directions are descriptive only" in protocol


def test_chronological_failure_boundary_precedes_reverse(monkeypatch: pytest.MonkeyPatch) -> None:
    calls=[]
    def fake_run(root: Path, dataset: str, reverse: bool=False):
        calls.append(reverse)
        if reverse:
            raise module.NumericalContractError("reverse only")
        return {"status":"chronological complete"}
    monkeypatch.setattr(module,"run_dataset_observed",fake_run)
    monkeypatch.setattr(module,"load_config",lambda root: ({"project":{"output_dir":"out"}},"hash"))
    monkeypatch.setattr(module,"atomic_write_json",lambda *args,**kwargs: None)
    result=module.run_stieger_observed(Path("."))
    assert calls==[False,True]
    assert result["chronological"]["status"]=="chronological complete"
    assert result["reverse_non_voting"]["status"]=="CONTROL_UNASSESSED_OPTIMIZATION_FAILURE"
