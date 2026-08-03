def test_top_level_exports():
    import lerobot_policy_rewind_state_ours_variant8 as m
    assert hasattr(m, "RewindStateOursVariant8Config")
    assert hasattr(m, "RewindStateOursVariant8Policy")
    assert hasattr(m, "make_rewind_state_ours_variant8_pre_post_processors")
    assert hasattr(m, "FrozenDINOv2Encoder")
    assert hasattr(m, "FrozenMiniLMTextEncoder")
