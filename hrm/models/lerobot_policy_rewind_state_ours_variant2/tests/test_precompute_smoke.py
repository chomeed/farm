from __future__ import annotations

import sys

import pytest


def test_precompute_module_imports_and_has_main():
    from lerobot_policy_rewind_state_ours_variant2.common import precompute_preference_pairs as m
    assert hasattr(m, "main")
    assert hasattr(m, "parse_args")


def test_precompute_argparser_smoke(monkeypatch):
    from lerobot_policy_rewind_state_ours_variant2.common import precompute_preference_pairs as m
    monkeypatch.setattr(
        sys, "argv",
        ["precompute", "--base_ckpt", "x", "--success_repo_id", "a",
         "--failure_repo_id", "b", "--output_manifest", "/tmp/o.json"],
    )
    args = m.parse_args()
    assert args.base_ckpt == "x"
    assert args.success_repo_id == "a"
    assert args.failure_repo_id == "b"
    assert args.output_manifest == "/tmp/o.json"
    assert args.clip_len == 16
    assert args.frame_stride == 1
    assert args.smooth_window == 7
    assert args.top_k == 3
