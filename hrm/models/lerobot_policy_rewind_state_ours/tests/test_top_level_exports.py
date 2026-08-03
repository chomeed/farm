def test_top_level_exports():
    import lerobot_policy_rewind_state_ours as m
    assert hasattr(m, "RewindStateOursConfig")
    assert hasattr(m, "RewindStateOursPolicy")
    assert hasattr(m, "make_rewind_state_ours_pre_post_processors")
    assert hasattr(m, "FrozenDINOv2Encoder")
    assert hasattr(m, "FrozenMiniLMTextEncoder")
