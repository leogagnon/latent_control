from abc import ABC
from typing import *
from models.mamba import MambaLMHeadModel, MambaConfig
from models.gpt import GPT, GPTConfig
import torch.nn as nn
import torch

class DecoderModel(ABC, nn.Module):
    def forward(self, input_ids, context_enc):
        pass

class MambaDecoder(DecoderModel, MambaLMHeadModel):
    def __init__(self, config: Optional[MambaConfig] = None, **kwargs):
        if config is None:
            config = MambaConfig(**kwargs)
        super().__init__(config)

    def forward(
        self,
        input_ids,
        context_enc,
        inference_params=None,
        only_last_logits=False,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        # TODO : give hidden state as context encoding
        if context_enc is not None:
            raise NotImplementedError

        hidden_states = self.wte(input_ids)

        hidden_states = self.backbone(
            input_ids, inference_params=inference_params, **mixer_kwargs
        )
        if only_last_logits:
            hidden_states = hidden_states[:, -1]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
    
class GPTDecoder(DecoderModel, GPT):
    def __init__(self, config: Optional[GPTConfig] = None, **kwargs):
        if config is None:
            config = GPTConfig(**kwargs)
        super().__init__(config)

    def forward(
        self,
        input_ids,
        context_enc,
        attn_mask=None,
        only_last_logits=False,

    ):
        device = input_ids.device
        b, t = input_ids.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            input_ids
        )  # token embeddings of shape (b, t, n_embd)

        if self.config.positional_encodings:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            tok_emb = tok_emb + pos_emb

        x = self.transformer.drop(tok_emb)

        if context_enc != None:
            x = torch.concatenate([context_enc, x], dim=-2)

        for block in self.transformer.h:
            x = block(x, attn_mask)
        x = self.transformer.ln_f(x)

        if context_enc != None:
            x = x[:,1:]

        if only_last_logits:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
        else:
            logits = self.lm_head(x)

        return logits