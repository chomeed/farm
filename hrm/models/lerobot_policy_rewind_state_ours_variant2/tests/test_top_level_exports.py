def test_top_level_exports():
    import lerobot_policy_rewind_state_ours_variant2 as m
    assert hasattr(m, "RewindStateOursVariant2Config")
    assert hasattr(m, "RewindStateOursVariant2Policy")
    assert hasattr(m, "make_rewind_state_ours_variant2_pre_post_processors")
    assert hasattr(m, "FrozenDINOv2Encoder")
    assert hasattr(m, "FrozenMiniLMTextEncoder")
