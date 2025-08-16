from typing import Optional, Union

import numpy as np
import torch

from src.model.symbolic_converter import build_env
from symbolicregression.envs.environment import FunctionEnvironment
from symbolicregression.envs.generators import Node
from symbolicregression.model.embedders import LinearPointEmbedder
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.model.transformer import TransformerModel

_model = None


def build_symbolic_model() -> ModelWrapper:
    global _model
    if _model is None:
        _model = torch.load("./symbolicregression/weights/model.pt", weights_only=False)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # _model = _model.to(device)
    return _model


@torch.no_grad()
def derive_embedding(
    x_to_fit: Union[torch.Tensor, np.ndarray],
    y_to_fit: Union[torch.Tensor, np.ndarray],
    token_pooling: str = "mean",
    bag_aggregation: str = "mean",
    env: FunctionEnvironment = build_env(),
    tree_encoded: Optional[Node] = None,
    model: ModelWrapper = build_symbolic_model(),
    max_seq_length: int = 200,
) -> torch.Tensor:
    assert token_pooling in ["mean", "max", "last"]
    assert bag_aggregation in ["mean", "max", "last"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder: LinearPointEmbedder = model.embedder
    encoder: TransformerModel = model.encoder
    decoder: TransformerModel = model.decoder

    embedder.eval()
    encoder.eval()
    decoder.eval()

    if tree_encoded is not None:
        embedder.to(device)
        encoder.to(device)
        decoder.to(device)

        if len(y_to_fit.shape) == 1:
            y_to_fit = y_to_fit.reshape(-1, 1)

        seq_length = len(x_to_fit)
        num_bags = (seq_length + max_seq_length - 1) // max_seq_length

        bag_embeddings = []
        for bag_id in range(num_bags):
            start_idx = bag_id * max_seq_length
            end_idx = min((bag_id + 1) * max_seq_length, seq_length)

            # handle current bag
            x1 = [[]]
            for seq_l in range(start_idx, end_idx):
                x1[0].append([x_to_fit[seq_l], y_to_fit[seq_l]])

            x1, len1 = embedder(x1)

            x2, len2 = env.batch_equations(
                env.word_to_idx([tree_encoded], float_input=False)
            )

            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1

            x2 = x2[: len2.max()]
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            x2 = x2.to(device)
            len2 = len2.to(device)
            y = y.to(device)

            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
            )

            # pooling on token level over the sequence
            if token_pooling == "last":
                bag_embedding = decoded[-1, 0, :]
            elif token_pooling == "mean":
                bag_embedding = decoded[:, 0, :].mean(dim=0)
            elif token_pooling == "max":
                bag_embedding = decoded[:, 0, :].max(dim=0)[0]

            bag_embeddings.append(bag_embedding)

        # aggregate embeddings of all bags
        if len(bag_embeddings) > 1:
            bag_embeddings = torch.stack(bag_embeddings)
            if bag_aggregation == "last":
                final_embedding = bag_embeddings[-1]
            elif bag_aggregation == "mean":
                final_embedding = bag_embeddings.mean(dim=0)
            elif bag_aggregation == "max":
                final_embedding = bag_embeddings.max(dim=0)[0]
        else:
            final_embedding = bag_embeddings[0]

    else:
        raise NotImplementedError

    return final_embedding.reshape(1, -1).detach().cpu()
