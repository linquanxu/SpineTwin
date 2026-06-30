import torch
from torch import nn
from model.Base_module import TransformerEncoderLayer, VideoBaseEmbedding,TokenBaseEmbedding
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

try:
    from tools import norm_state as _norm_state
except Exception:
    def _norm_state(x):  
        return x

def _gate_reached(pk_val, gate):
    return (gate is not None) and (pk_val >= gate)
class BertPredictionHeadTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.transform_act_fn = nn.GELU()

        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLMHead(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform()
        self.decoder = nn.Linear(768, 49411, bias=False)
        self.bias = nn.Parameter(torch.zeros(49411))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        layers = []
        dpr = [0.1 for _ in range(12)]
        for layer_idx in range(12):

            layers.append(
                TransformerEncoderLayer(
                    d_model=768,
                    nhead=12,
                    dim_feedforward=3072,
                    dropout=0.,
                    drop_path_ratio=dpr[layer_idx],
                    activation="gelu",
                    layer_scale=True,
                    ls_init_values=1e-3,
                    batch_first=True,
                    norm_first=True,
                ))
        self.layers = nn.ModuleList(
            layers
        )

    def forward(self, data, mask=None):
        feature_list = []
        for l, layer_module in enumerate(self.layers):
            data = layer_module(src=data, src_mask=mask)
            feature_list.append(data)
        return data, feature_list