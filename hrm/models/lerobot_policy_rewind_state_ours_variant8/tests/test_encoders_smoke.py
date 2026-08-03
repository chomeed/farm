def test_encoders_importable():
    from lerobot_policy_rewind_state_ours_variant8.vision_encoder import FrozenDINOv2Encoder
    from lerobot_policy_rewind_state_ours_variant8.text_encoder import FrozenMiniLMTextEncoder
    assert FrozenDINOv2Encoder is not None
    assert FrozenMiniLMTextEncoder is not None
