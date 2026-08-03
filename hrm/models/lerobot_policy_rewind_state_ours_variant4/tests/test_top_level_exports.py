def test_top_level_exports():
    import lerobot_policy_rewind_state_ours_variant4 as m
    assert hasattr(m, "RewindStateOursVariant4Config")
    assert hasattr(m, "RewindStateOursVariant4Policy")
    assert hasattr(m, "make_rewind_state_ours_variant4_pre_post_processors")
    assert hasattr(m, "FrozenDINOv2Encoder")
    assert hasattr(m, "FrozenMiniLMTextEncoder")
