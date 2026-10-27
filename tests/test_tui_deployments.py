# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deployments in the TUI: telling them apart, the list's two views, the preview and the job screen's tabs."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

from cortex_training.tui import views as v

NOW = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
UUID = "00000000-0000-4000-8000-000000000008"  # a made-up canonical UUID
DEPLOY = "8:sampling:0"


def _deployment(status="RUNNING", job_id="8", cls="EVALUATION", **extra):
    return {"job_id": job_id, "status": status, "idle_timeout_seconds": 0, **extra,
            "sub_jobs": [{"sub_job_id": f"{job_id}:sampling:0", "job_type": "sampling", "status": status,
                          "model_name": "org/model",
                          "inference_config": {"deployment_class": cls, "n_gpus": 1.0, "num_replicas": 2.0}}]}


def _sampling():
    return {"job_id": "9", "status": "SUCCEEDED",
            "sub_jobs": [{"sub_job_id": "9:sampling:0", "job_type": "sampling", "inference_config": {"n_gpus": 1.0}}]}


def _training():
    return {"job_id": "7", "status": "RUNNING",
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"},
                         {"sub_job_id": "7:sampling:0", "job_type": "sampling", "inference_config": {}}]}


def test_the_cli_s_rule_tells_a_deployment_apart():
    assert v.is_deployment(_deployment())
    assert not v.is_deployment(_deployment(cls="PRODUCTION"))  # not one until the CLI's rule says so
    assert not v.is_deployment(_sampling())
    assert not v.is_deployment(_training())
    assert not v.is_deployment({"job_id": "1"})  # an old record without sub-jobs stays a job
    assert not v.is_deployment({"job_id": "1", "sub_jobs": [{"inference_config": None}]})
    assert not v.is_deployment("not a job")
    assert v.deployment_class(_deployment()) == "EVALUATION"
    assert v.deployment_class(_training()) == ""
    assert v.deployment_sub_job_id(_deployment()) == DEPLOY
    assert v.deployment_sub_job_id(_training()) is None


def test_a_deployment_s_status_reads_in_deployment_words():
    assert v.status_word_for(_deployment(), "RUNNING") == "ready"
    assert v.status_word_for(_deployment(), "JOB_STATE_PLACING") == "initializing"
    assert v.status_word_for(_deployment(), "CANCELLED") == "dropped"
    assert v.status_word_for(_deployment(), "") == "unknown"
    assert v.status_word_for(_training(), "RUNNING") == "running"
    assert v.status_word_for(_training(), "") == "?"


def test_a_deployment_reads_deployment_in_the_type_column_and_search():
    job = _deployment()
    assert v.job_type_label(job) == "deployment"
    assert v.job_type_label(_sampling()) == "sampling"
    assert v.job_matches_query(job, v.parse_query("type:deployment"))
    assert not v.job_matches_query(_sampling(), v.parse_query("type:deployment"))
    assert v.job_matches_query(job, v.parse_query("class:evaluation"))
    assert not v.job_matches_query(_training(), v.parse_query("class:evaluation"))


def test_preview_facts_of_a_deployment():
    facts = dict(v.preview_facts(_deployment(job_id=UUID), NOW, base_url="https://HOST/x/inference/v1"))
    assert facts["Type"] == "deployment"
    assert facts["Class"] == "EVALUATION"
    assert facts["Replicas"] == "2"
    assert facts["Idle timeout"] == "off"  # 0 disables reclamation, it is not a zero-second timeout
    assert facts["Call as"] == f"sf-ct-{UUID}"
    assert facts["Base URL"] == "https://HOST/x/inference/v1"
    assert dict(v.preview_facts(_deployment(idle_timeout_seconds=1800), NOW))["Idle timeout"] == "30m 00s"
    assert "Base URL" not in dict(v.preview_facts(_deployment(), NOW))
    # The client's helper refuses an ID that isn't a canonical UUID; the TUI then names no model.
    assert "Call as" not in dict(v.preview_facts(_deployment(job_id="8"), NOW))
    assert "Class" not in dict(v.preview_facts(_training(), NOW))


def test_base_url_comes_from_the_client():
    class Client:
        model_deployment_base_url = "https://HOST/api/v2/databases/DB/schemas/PUBLIC/cortex-training/inference/v1"

    assert v.base_url(Client()) == Client.model_deployment_base_url
    assert v.base_url(object()) == ""
