import torch
import torch.nn as nn
from src.vpr.feature.api import AudioEncoder
from src.vpr.feature import pooling_v2


class Audio2Vec_based_Adapter(nn.Module):
    def __init__(
        self,
        model_name='openai/whisper-tiny', 
        frozen_encoder=True,
        bnb_config=None,
        peft_config=None,
        encoder_config=None,
        n_mfa_layers=1,
        pooling_layer='ASP', 
        embd_dim=256,
        adapter_dim=128,
        dropout=0,
        ):
        super(Audio2Vec_based_Adapter, self).__init__()

        print("model_name", model_name)
        print("frozen_encoder", frozen_encoder)
        print("bnb_config", bnb_config)
        print("peft_config", peft_config)
        print("encoder_config", encoder_config)

        self.front = AudioEncoder(
            model_name, 
            frozen_encoder,
            bnb_config,
            peft_config,
            encoder_config,
            )
        self.drop = nn.Dropout(dropout) if dropout else None

        if n_mfa_layers == -1:
            self.n_mfa_layers = self.front.n_hidden_states
        else:
            self.n_mfa_layers = n_mfa_layers
        assert 1 <= self.n_mfa_layers <= self.front.n_hidden_states, \
            'Invalid Input: n_mfa_layers'
        
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.front.d_model, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(True),
                nn.Linear(adapter_dim, adapter_dim),
            ) for _ in range(self.n_mfa_layers)
        ])

        
        feat_dim = adapter_dim * self.n_mfa_layers
        
        self.pooling = getattr(pooling_v2, pooling_layer)(
            feat_dim, 
            adapter_dim,
            )
        
        self.bottleneck = nn.Linear(
            feat_dim * self.pooling.expansion, 
            embd_dim,
            )

    def forward(self, x):
        x = self.front(x)
        if self.n_mfa_layers == 1:
            x = x.last_hidden_state
        else:
            layer_outputs = []
            x.hidden_states = x.hidden_states[-self.n_mfa_layers:]
            for i in range(self.n_mfa_layers):
                layer_outputs.append(self.adapter_layers[i](x.hidden_states[i]))
            x = torch.cat(layer_outputs, dim=-1)
            
        x = self.pooling(x)    
        if self.drop: 
            x = self.drop(x)
        x = self.bottleneck(x)

        return x
