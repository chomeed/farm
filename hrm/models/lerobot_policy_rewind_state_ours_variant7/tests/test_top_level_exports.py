def test_top_level_exports():
    import lerobot_policy_rewind_state_ours_variant7 as m
    assert hasattr(m, "RewindStateOursVariant7Config")
    assert hasattr(m, "RewindStateOursVariant7Policy")
    assert hasattr(m, "make_rewind_state_ours_variant7_pre_post_processors")
    assert hasattr(m, "FrozenDINOv2Encoder")
    assert hasattr(m, "FrozenMiniLMTextEncoder")
