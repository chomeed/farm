import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel, AutoModel, AutoImageProcessor
from typing import List
from PIL import Image

class FrozenDINOv2Encoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(ckpt).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_image(
        self,
        images: List[Image.Image],
        do_rescale: bool = False,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor:
        """
        images: list[PIL.Image], length B

        returns:
          - (B, hidden_dim) if return_patch_tokens=False
              uses pooled CLS embedding
          - (B, num_patches, hidden_dim) if return_patch_tokens=True
              uses patch tokens only, excluding CLS
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=do_rescale,
        ).to(self.device)

        outputs = self.model(**inputs)

        if return_patch_tokens:
            # last_hidden_state shape: (B, 1 + num_patches, hidden_dim)
            # token 0 is CLS, remaining are patch tokens
            return outputs.last_hidden_state[:, 1:, :]
        else:
            # pooled CLS representation
            return outputs.pooler_output

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(ckpt).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)

        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, 512) CLIP text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        return text_embeds

    def encode_image(self, images: List[Image.Image], do_rescale=False) -> torch.Tensor:
        """
        images: list of PIL Images, length B
        returns: (B, 512) CLIP image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = FrozenCLIPEncoder("openai/clip-vit-base-patch32", device)
    encoder = FrozenDINOv2Encoder("facebook/dinov2-base", device)

    # Test text encoding
    # texts = ["a photo of a cat", "a photo of a dog"]
    # text_embeds = encoder.encode_text(texts)
    # print("Text embeddings shape:", text_embeds.shape)  # Should be (2, 512)

    # Test image encoding
    images = [Image.new("RGB", (224, 224), color="red")]
    image_embeds = encoder.encode_image(images)
    print("Image embeddings shape:", image_embeds.shape)  # Should be (1, 512) for CLIP emb or (1, 768) for DINOv2 img emb