import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List


class FrozenMiniLMTextEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device, normalize: bool = True):
        super().__init__()
        self.device = device
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden_state: (B, T, D)
        attention_mask: (B, T)
        returns: (B, D)
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode_text(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, hidden_dim)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        text_embeds = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

        if self.normalize:
            text_embeds = F.normalize(text_embeds, p=2, dim=1)

        return text_embeds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = FrozenMiniLMTextEncoder(
        ckpt="sentence-transformers/all-MiniLM-L12-v2",
        device=device,
        normalize=True,
    )

    texts = [
        "pick up the red block",
        "open the drawer",
        "move the cup to the shelf",
    ]

    text_embeds = encoder.encode_text(texts)
    print("Text embeddings shape:", text_embeds.shape)