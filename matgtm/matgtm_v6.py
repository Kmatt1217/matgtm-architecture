import math
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from transformers import pipeline
from torchvision import models
from fairseq.optim.adafactor import Adafactor
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError, MeanAbsoluteError
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import os
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import time
import pandas as pd
from typing import Optional
from pyts.image import GramianAngularField



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) /d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)
    
class Sinusoidal2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        """
        Args:
            d_model (int): The dimension of the model (must be even).
            height (int): Height of the input image/grid.
            width (int): Width of the input image/grid.
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D positional encoding.")

        self.d_model = d_model
        height = int(height)
        width = int(width)
        self.height = int(height)
        self.width = int(width)

        pe = torch.zeros(d_model, height, width)

        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))

        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)

        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)

        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, C, H, W)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            x + positional_encoding
        """
        # return x + self.pe[:, :, :x.size(2), :x.size(3)]
        return x + self.pe[:, :, :x.size(-2), :x.size(-1)]

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1,x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0),-1,y.size(-1))
        else:
            y = y.view(-1,x.size(1),y.size(-1))

        return y

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super(FusionNetwork, self).__init__()
        
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text
        input_dim = embedding_dim + (embedding_dim*use_img) + (embedding_dim*use_text)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        # Build input
        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img) 
        if self.use_text == 1:
            decoder_inputs.append(text_encoding) 
        decoder_inputs.append(dummy_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)

        final = self.feature_fusion(concat_features)
        # final = self.feature_fusion(dummy_encoding)

        return final

class GAFEncoder:
    def __init__(self, image_size:int=32,gaf_method='summation'):
        self.image_size = image_size
        self.gaf = GramianAngularField(image_size=image_size, method=gaf_method)

    def transform(self, batch_ts):
        batch_ts = batch_ts.detach().cpu().numpy()
        gaf_images = []
        for sample in batch_ts:

            channels = [self.gaf.fit_transform(channel.reshape(1,-1))[0] for channel in sample]
            gaf_img = np.stack(channels, axis=0)
            gaf_images.append(gaf_img)

        gaf_images = np.stack(gaf_images,axis=0)
        return torch.FloatTensor(gaf_images)

class VisionEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.proj = nn.Linear(2048, out_dim)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  
        
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)                  # [B, 2048, H/32, W/32]
        x = self.adaptive_pool(x)           # [B, 2048, 8, 8]
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 64, 2048]
        x = self.proj(x)                    # [B, 64, out_dim]
        return x
        
class LatentQueryPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embedding = nn.Parameter(torch.randn(max_len, 1, embedding_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.size()

        if seq_len > self.pos_embedding.size(0):
            raise ValueError(f"Latent query token length ({seq_len}) exceeds max_len ({self.pos_embedding.size(0)})")

        return x + self.pos_embedding[:seq_len]  

class GTrendQueryDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforawrd=2048 ,dropout=0.1, activation='relu',max_len=256,use_mask=1):
        super().__init__()
        
        self.nhead = nhead
        self.use_mask = use_mask
        # self.latent_query_pos_embed = LatentQueryPositionalEncoding(d_model, max_len)
        
        self.self_attn = nn.MultiheadAttention(d_model, self.nhead, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, self.nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforawrd)
        self.dropout= nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforawrd, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = F.relu
        
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(GTrendQueryDecoderLayer, self).__setstate__(state)
    
    def forward(self, latent_query_token, gtrend_emb, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, 
            memory_key_padding_mask = None, tgt_is_causal=None, memory_is_causal=None):

        if gtrend_emb.ndim == 4:
            # Expecting shape [B, C, H, W]
            B, C, H, W = gtrend_emb.shape
            gtrend_emb = gtrend_emb.view(B, C, H * W).permute(2, 0, 1)  # [L, B, E]

        elif gtrend_emb.ndim != 3:
            raise ValueError(f"Expected 3D (L,B,E) or 4D (B,C,H,W), got {gtrend_emb.shape}")
         # Step 1: Self-Attention on Google Trend Embeddings
        if self.use_mask == 1:
            gtrend_self_attn_output, _ = self.self_attn(gtrend_emb, gtrend_emb,gtrend_emb,attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask)
        else:
            gtrend_self_attn_output, _ = self.self_attn(gtrend_emb, gtrend_emb, gtrend_emb)
            
        gtrend_self_attn_token = gtrend_self_attn_output + self.dropout2(gtrend_emb)
        gtrend_self_attn_token = self.norm2(gtrend_self_attn_token)
        
        # Step 2: Cross-Attention with Latent Query Tokens
        if isinstance(latent_query_token, tuple):
            latent_query_token = latent_query_token[0]
        
        # latent_query_token = self.latent_query_pos_embed(latent_query_token)
        latent_query_token = self.norm1(latent_query_token)


        tgt, attn_weights = self.multihead_attn(latent_query_token, gtrend_self_attn_token, gtrend_self_attn_token)
        tgt = latent_query_token + self.dropout3(tgt)
        tgt = self.norm3(tgt)

        # Step 3: Feed-Forward Network (FFN)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, attn_weights
    
class GAFGTrendEmbedder(nn.Module):
    def __init__(self, image_size, visionencoder_out_dim,
                 embedding_dim, hidden_dim, M, gaf_method, gpu_num, use_mask,weight_init='xavier', enc_num_layers:int=2):
        super().__init__()

        self.M = M
        self.gaf_encoder = GAFEncoder(image_size = image_size,gaf_method=gaf_method)
        self.vision_encoder = VisionEncoder(out_dim=visionencoder_out_dim)
        self.latent_queries = nn.Parameter(torch.randn(M, embedding_dim))
        if weight_init =='xavier':
            nn.init.xavier_uniform_(self.latent_queries)
        elif weight_init == 'normal':
            nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        query_trend_layer = GTrendQueryDecoderLayer(d_model=embedding_dim,nhead=4,dropout=0.2,use_mask=use_mask)
        # self.decoder = nn.TransformerDecoder(query_trend_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(query_trend_layer, num_layers=enc_num_layers)
        self.grid_proj = nn.Linear(visionencoder_out_dim, embedding_dim)
        self.gpu_num = gpu_num
        self.memory_proj = nn.Linear(embedding_dim, hidden_dim)
        self.use_mask = use_mask
        self.pos_encoder = Sinusoidal2DPositionalEncoding(d_model=embedding_dim, height=image_size, width=image_size)
    
    def generate_allowed_m(self):
        allowed = []
        current = self.M
        while current >= 2:
            allowed.append(current)
            current //= 2
        return allowed
        
    def _get_gaf_padding_mask(self, gaf_images: torch.Tensor, threshold: float = 0.01):
        B, C, H, W = gaf_images.shape
        avg_intensity = gaf_images.abs().mean(dim=1) # [B, H, W]
        mask = (avg_intensity < threshold).view(B,-1) # [B,H*W]
        
        masked_ratio_per_sample = mask.float().mean(dim=1)
        avg_mask_ratio = masked_ratio_per_sample.mean().item()
        print(f"[Masking] Threshold = {threshold:.4f}, Avg masking ratio = {avg_mask_ratio * 100:.2f}%")
        return mask
        
    def forward(self, gtrends, m=None):
        device = gtrends.device
        batch_size = gtrends.size(0)

        gaf_images= self.gaf_encoder.transform(gtrends).to(device) # [B, num_channel(3), H(image_size), W(image_size)]
        if self.use_mask == 1:
            memory_key_padding_mask = self._get_gaf_padding_mask(gaf_images)
        else:
            memory_key_padding_mask = None
            
        grid_map = self.vision_encoder(gaf_images) # [B, HW, visionencoder_out_dim]
        grid_map = self.grid_proj(grid_map) # [B, HW, embedding_dim]
        grid_map = grid_map.view(batch_size, grid_map.shape[-1], int(np.sqrt(grid_map.shape[1])), -1)
        grid_map = self.pos_encoder(grid_map)  
        
 
        max_queries = self.latent_queries.size(0)
        if m is None:
            if self.training:  # pytorch-lightning will set this automatically
                allowed_m = self.generate_allowed_m()
                m = np.random.choice(allowed_m, p=[1 / len(allowed_m)] * len(allowed_m))
            else:
   
                m = self.

        

        latent_q = self.latent_queries[:m].to(device).unsqueeze(1).expand(-1, batch_size, -1)          
        grid_feats = grid_map.view(batch_size, self.latent_queries.shape[1], -1).permute(2, 0, 1)  # [HW, B, D]

        query = latent_q  # [m, B, D]
        key = grid_feats  # [HW, B, D]
        similarity = torch.matmul(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2))  # [B, m, HW]
        
        tokens, _ = self.decoder(latent_q, grid_feats,memory_key_padding_mask=memory_key_padding_mask)
        tokens = self.memory_proj(tokens)

        return tokens, m
        

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, M, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.M = M
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        self.latent_queries = nn.Parameter(torch.randn(M, embedding_dim))
        query_trend_layer = GTrendQueryDecoderLayer(d_model=embedding_dim,nhead=4,dropout=0.2)
        self.decoder = nn.TransformerDecoder(query_trend_layer,num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num
        
    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask
    
    def generate_allowed_m(self):
        allowed = []
        current = self.M
        while current >= 2:
            allowed.append(current)
            current //= 2
        return allowed
    
    def forward(self, gtrends, m=None):
        device = gtrends.device
        gtrends = gtrends.float()
        batch_size = gtrends.size(0)
        gtrend_emb = self.input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.pos_embedding(gtrend_emb.transpose(0,1))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        
        if m is None:
            allowed_m = self.generate_allowed_m()
            m = np.random.choice(allowed_m, p=[1/len(allowed_m)] * len(allowed_m))
            
        latent_q = self.latent_queries[:m].to(device).unsqueeze(1).expand(-1, batch_size, -1)

        if self.use_mask == 1:
            tokens = self.decoder(latent_q, gtrend_emb, memory_mask = input_mask)
        else:
            tokens = self.decoder(latent_q, gtrend_emb)
            
        
        return tokens, m

class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.word_embedder = pipeline('feature-extraction', model='bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, category, color, fabric):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        textual_description = [self.col_dict[color.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.fab_dict[fabric.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.cat_dict[category.detach().cpu().numpy().tolist()[i]] for i in range(len(category))]


        # Use BERT to extract features
        word_embeddings = self.word_embedder(textual_description)

        # BERT gives us embeddings for [CLS] ..  [EOS], which is why we only average the embeddings in the range [1:-1] 
        # We're not fine tuning BERT and we don't want the noise coming from [CLS] or [EOS]
        word_embeddings = [torch.FloatTensor(x[0][1:-1]).mean(axis=0) for x in word_embeddings] 
        word_embeddings = torch.stack(word_embeddings).to(device)

        
        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings)).to(device)

        return word_embeddings
    
class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        
    def forward(self, images):        
        img_embeddings = self.resnet(images)  
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2],-1)

        return out.view(*size).contiguous() # batch_size, 2048, image_size/32, image_size/32

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout = nn.Dropout(0.2)


    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings
    
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True):
        
        super().__init__(d_model=d_model, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, layer_norm_eps=layer_norm_eps,
                         batch_first=batch_first, norm_first=norm_first, bias=bias)
        
        self.granularity_scale = 1.0
        self.dim_feedforward = dim_feedforward
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False,\
                    inference_granularity=None):
        
        if isinstance(tgt, tuple):
            tgt = tgt[0]
        if isinstance(memory, tuple):
            memory = memory[0]
            
        chosen_dim = int(self.dim_feedforward * self.granularity_scale)
        self.chosen_dim = chosen_dim
        
        if self.norm_first:
            tgt = tgt + self._sa_block(self.norm1(tgt),tgt_mask,tgt_key_padding_mask, tgt_is_causal)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory,memory_mask, memory_key_padding_mask, memory_is_causal)
            tgt = tgt + self._ff_block(self.norm3(tgt),chosen_dim)
            
        else:
            tgt = self.norm1(tgt + self._sa_block(tgt,tgt_mask,tgt_key_padding_mask, tgt_is_causal))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory,memory_mask, memory_key_padding_mask, memory_is_causal))
            tgt = self.norm3(tgt + self._ff_block(tgt, chosen_dim))
                             
        return tgt
            
    def _sa_block(self, x, attn_mask, key_padding_mask,is_causal: bool = False):
        x, _ = self.self_attn(x,x,x,attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)
        return self.dropout1(x)
    
    def _mha_block(self, x, memory, attn_mask, key_padding_mask,is_causal: bool = False):
        x, _ = self.multihead_attn(x, memory, memory, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)
        return self.dropout2(x)
        
    def _ff_block(self, x , chosen_dim):
        
        ffn_input = F.linear(x, self.linear1.weight[:chosen_dim,:],self.linear1.bias[:chosen_dim])
        ffn_intermediate = self.dropout(self.activation(ffn_input))
        ffn_output = F.linear(ffn_intermediate,self.linear2.weight[:,:chosen_dim],self.linear2.bias[:chosen_dim])
        return self.dropout3(ffn_output)
        
        
    
class MatFormerModel(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, num_layers, granularity_scales):
        super(MatFormerModel, self).__init__()
        
        self.granularity_scales = granularity_scales
        decoder_layer = CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.num_layers = num_layers
  

    def forward(self, tgt, memory, granularity_scales=None):
        for i, mod in enumerate(self.decoder.layers):
            if self.training:
                # Randomly choose a granularity scale for each layer
                chosen_granularity = random.choice(self.granularity_scales)
                mod.granularity_scale = chosen_granularity
                print(f"Layer {i} - chosen granularity scale (training): {chosen_granularity}")
            else:
                # In inference mode, use the passed granularity_scales if provided
                if granularity_scales is not None:
                    mod.granularity_scale = granularity_scales[i]
                    print(f"Layer {i} - chosen granularity scale (inference): {granularity_scales[i]}")
            tgt = mod(tgt, memory)
            
        return tgt
        
class MatGTM(pl.LightningModule):
    def __init__(self, args, embedding_dim, hidden_dim, output_dim, num_heads,\
                 enc_num_layers,dec_num_layers, use_text, use_img, cat_dict, col_dict, fab_dict, trend_len,\
                 num_trends, gpu_num, \
                 granularity_scales:list=[1,1/2,1/4,1/8],
                 selected_m:int=None,
                 use_encoder_mask=1, autoregressive=False,
                 gaf='False',
                 gaf_image_size:int=32,
                 gaf_method='summation',
                 weight_init='xavier',
                 visionencoder_out_dim:int=256
                 ):
        super().__init__()
        
        self.args = args
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.save_hyperparameters()

        self.M=args.M
        self.test_results = []

        self.training_logs = []
        self.validation_logs = []
        self.test_logs = []
        
        # for inferecne
        self.selected_m = selected_m

        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict,gpu_num)
        self.image_encoder = ImageEmbedder()
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        if gaf is False:
            self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends,M=self.M, gpu_num=self.gpu_num)

        else:
            self.gtrend_encoder = GAFGTrendEmbedder(gaf_image_size, visionencoder_out_dim,embedding_dim, hidden_dim, M=self.M, gaf_method=gaf_method, gpu_num=self.gpu_num, 
                                                    use_mask = use_encoder_mask,weight_init = weight_init,enc_num_layers=enc_num_layers)
        
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim,
                                            use_img, use_text)

        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        if self.autoregressive:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
            
        self.decoder = MatFormerModel(d_model=self.hidden_dim,
                                      dim_feedforward=self.hidden_dim * 4,
                                      nhead=num_heads,
                                      num_layers=dec_num_layers,
                                      granularity_scales=granularity_scales)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(args.hidden_dim, self.output_len), 
            nn.Dropout(0.2))

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(f'cuda:{self.gpu_num}')
        
    def forward(self, category, color, fabric, temporal_features, gtrends, images,\
                selected_m=None, granularity_scales=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_encoding = self.image_encoder(images).to(device)
        dummy_encoding = self.dummy_encoder(temporal_features).to(device)
        text_encoding = self.text_encoder(category, color, fabric).to(device)
        
        m = selected_m if selected_m is not None else self.selected_m

        gtrend_encoding, chosen_m  = self.gtrend_encoder(gtrends.to(device),m)
        self.chosen_m = chosen_m
        
        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)
        
        tgt = static_feature_fusion.unsqueeze(0)
        memory = gtrend_encoding

        if granularity_scales is not None:
            decoder_out = self.decoder(tgt, memory, granularity_scales=granularity_scales)
        else:
            decoder_out = self.decoder(tgt, memory)
            
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len)

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images = train_batch 
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        
        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065
        
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())

        mae, wmape, adj_smape, accum_smape = self.calculate_metrics(rescaled_item_sales, rescaled_forecasted_sales)
           
        self.log('train_mae', mae, on_step=True, on_epoch=True,  logger=True,prog_bar=True)
        self.log('train_wmape', wmape, on_step=True, on_epoch=True,  logger=True,prog_bar=True)
        self.log('train_ad_smape', adj_smape, on_step=True, on_epoch=True,  logger=True,prog_bar=True)
        self.log('train_accum_smape', accum_smape, on_step=True, on_epoch=True,  logger=True,prog_bar=True)
        self.log('train_loss_mse', loss, on_step=True, on_epoch=True,  logger=True,prog_bar=True)
             
        chosen_m = getattr(self,'chosen_m',None)

        granularity_set = {}
        for idx, layer in enumerate(self.decoder.decoder.layers):
            if hasattr(layer, "chosen_granularity"):
                granularity_set[f"layer_{idx}"] = layer.chosen_granularity
                
        step_log = {
            "epoch" : self.current_epoch,
            "step" : batch_idx,
            "m" : chosen_m,
            "granularity_set": granularity_set,
            "mse":loss.item(),
            "mae": mae.item() if isinstance(mae, torch.Tensor) else mae,
            "wmape": wmape.item() if isinstance(wmape, torch.Tensor) else wmape,
            "ad_smape": adj_smape if isinstance(adj_smape, float) else float(adj_smape),
            "accum_smape": accum_smape.item() if isinstance(accum_smape, torch.Tensor) else accum_smape
        }
        self.training_logs.append(step_log)
     
        return loss
    
    def training_epoch_end(self,outputs):

        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, self.args.wandb_run+"_training_step_logs.csv")

        df = pd.DataFrame(self.training_logs)
        if os.path.exists(csv_file):
            df.to_csv(csv_file,mode='a',header=False,index=False)
        else:
            df.to_csv(csv_file,index=False)

        self.training_logs = []

    def validation_step(self, valid_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images = valid_batch 
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        rescaled_item_sales, rescaled_forecasted_sales = item_sales * 1065, forecasted_sales * 1065
          
        loss = F.mse_loss(item_sales, forecasted_sales)

        mae, wmape, adj_smape, accum_smape = self.calculate_metrics(rescaled_item_sales[:, :6], rescaled_forecasted_sales[:, :6])
        chosen_m = getattr(self,'chosen_m',None)

        granularity_set = {}
        for idx, layer in enumerate(self.decoder.decoder.layers):
            if hasattr(layer, "chosen_granularity"):
                granularity_set[f"layer_{idx}"] = layer.chosen_granularity
                
        step_log = {
            "epoch" : self.current_epoch,
            "step" : batch_idx,
            "m" : chosen_m,
            "granularity_set": granularity_set,
            "mse":loss.item(),
            "mae": mae.item() if isinstance(mae, torch.Tensor) else mae,
            "wmape": wmape.item() if isinstance(wmape, torch.Tensor) else wmape,
            "ad_smape": adj_smape if isinstance(adj_smape, float) else float(adj_smape),
            "accum_smape": accum_smape.item() if isinstance(accum_smape, torch.Tensor) else accum_smape
        }
        self.validation_logs.append(step_log)

        return item_sales.squeeze(), forecasted_sales.squeeze()
    
    def on_validation_epoch_start(self):
        self._epoch_start_time = time.time()

    def validation_epoch_end(self, val_step_outputs):
        epoch_time = time.time() - self._epoch_start_time
        self.log("val_epoch_time", epoch_time)

        item_sales = torch.cat([x[0] if x[0].dim() == 2 else x[0].unsqueeze(0) for x in val_step_outputs], dim=0)
        forecasted_sales = torch.cat([x[1] if x[1].dim() == 2 else x[1].unsqueeze(0) for x in val_step_outputs], dim=0)

        item_sales = item_sales
        forecasted_sales = forecasted_sales
        rescaled_item_sales, rescaled_forecasted_sales = item_sales*1065, forecasted_sales*1065
        
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        mae, wmape, adj_smape, accum_smape = self.calculate_metrics(rescaled_item_sales[:, :6], rescaled_forecasted_sales[:, :6])
    
        self.log('valid_mae', mae,  on_epoch=True,  logger=True,prog_bar=True)
        self.log('valid_wmape', wmape, on_epoch=True,  logger=True,prog_bar=True)
        self.log('valid_ad_smape', adj_smape, on_epoch=True,  logger=True,prog_bar=True)
        self.log('valid_accum_smape', accum_smape, on_epoch=True,  logger=True,prog_bar=True)
        self.log('valid_loss_mse', loss, on_epoch=True,  logger=True,prog_bar=True)
        
        output_dir = "result"
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, self.args.wandb_run+"_validation_step_logs.csv")

        df = pd.DataFrame(self.validation_logs)
        if os.path.exists(csv_file):
            df.to_csv(csv_file,mode='a',header=False,index=False)
        else:
            df.to_csv(csv_file,index=False)
      
        self.validation_logs = []

    def test_step(self, test_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images = test_batch 
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)

        rescaled_item_sales, rescaled_forecasted_sales = item_sales * 1065, forecasted_sales * 1065

        loss = F.mse_loss(item_sales, forecasted_sales)
        mae, wmape, adj_smape, accum_smape = self.calculate_metrics(rescaled_item_sales, rescaled_forecasted_sales)
        
        for i in range(len(images)):  
            self.test_results.append({
                # "id": key_id[i].item() if isinstance(key_id[i], torch.Tensor) else key_id[i],  
                "item_sales": item_sales[i].detach().cpu().numpy().tolist(),  
                "rescaled_item_sales": rescaled_item_sales[i].detach().cpu().numpy().tolist(),
                "rescaled_forecasted_sales": rescaled_forecasted_sales[i].detach().cpu().numpy().tolist()
                })

        
        
        chosen_m = getattr(self,'chosen_m',None)

        granularity_set = {}
        for idx, layer in enumerate(self.decoder.decoder.layers):
            if hasattr(layer, "chosen_granularity"):
                granularity_set[f"layer_{idx}"] = layer.chosen_granularity
                
        step_log = {
            "epoch" : self.current_epoch,
            "step" : batch_idx,
            "m" : chosen_m,
            "granularity_set": granularity_set,
            "mse":loss.item(),
            "mae": mae.item() if isinstance(mae, torch.Tensor) else mae,
            "wmape": wmape.item() if isinstance(wmape, torch.Tensor) else wmape,
            "ad_smape": adj_smape if isinstance(adj_smape, float) else float(adj_smape),
            "accum_smape": accum_smape.item() if isinstance(accum_smape, torch.Tensor) else accum_smape
        }

        self.test_logs.append(step_log)

        return item_sales, forecasted_sales

    def on_test_epoch_start(self):
        self._epoch_start_time = time.time()

    def test_epoch_end(self, test_step_outputs):

        epoch_time = time.time() - self._epoch_start_time
        self.log("test_epoch_time", epoch_time)

        item_sales = torch.cat([x[0] for x in test_step_outputs], dim=0)[:, :6]
        forecasted_sales = torch.cat([x[1] for x in test_step_outputs], dim=0)[:, :6]

        rescaled_item_sales, rescaled_forecasted_sales = item_sales * 1065, forecasted_sales * 1065

        mae, wape, rmae, rwape, \
         mae_25, mae_75, rmae_25, rmae_75, ad_smape = print_error_metrics(item_sales,
                                                                forecasted_sales,
                                                                rescaled_item_sales,
                                                                rescaled_forecasted_sales)
        chosen_m = getattr(self,'chosen_m',None)
        
        

        self.log('test_mae', mae, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_wape', wape, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_mae_25', mae_25, on_epoch=True, logger=True)
        self.log('test_mae_75', mae_75, on_epoch=True, logger=True)
        self.log('test_rescaled_mae', rmae, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_rescaled_wape', rwape, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_rescaled_mae_25', rmae_25, on_epoch=True, logger=True)
        self.log('test_rescaled_mae_75', rmae_75, on_epoch=True, logger=True)
        self.log('test_chosen_m',chosen_m,on_step=False,on_epoch=True,prog_bar=False,logger=True)
        self.log('test_ad_smape',ad_smape,on_epoch=True,logger=True)

        if hasattr(self, '_dummy_input'):
            dummy_input = self._dummy_input  # [category, color, fabric, temporal_features, gtrends, images]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            dummy_input = [x.to(device) for x in dummy_input]
            
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True, with_flops=True) as prof:
                with torch.inference_mode():
                    _ , _ = self.forward(*dummy_input, selected_m=self.selected_m)
            torch.cuda.synchronize()
            flops = sum([event.flops for event in prof.key_averages() if hasattr(event, 'flops')])
            self.log('test_flops', flops, on_epoch=True, logger=True)
            print(f"Test FLOPs: {flops}")
        else:
            print("No dummy input available for FLOPs calculation.")

    def calculate_metrics(self, predict, label):
        ad_smape = SymmetricMeanAbsolutePercentageError()
        mae_loss = MeanAbsoluteError()

        def wmape_func(label, predict):
            return (label - predict).abs().sum() / (label.sum() + 1e-10)

        if label.dim() == 1:
            label = label.unsqueeze(0)  # (output_dim,) → (1, output_dim)
        if predict.dim() == 1:
            predict = predict.unsqueeze(0)  # (output_dim,) → (1, output_dim)

            
        adj_smape = [ad_smape(label[i].detach().cpu(), predict[i].detach().cpu()) * 0.5 for i in range(label.size(0))]
        gt = torch.sum(label, dim=1).detach().cpu()
        pred = torch.sum(predict, dim=1).detach().cpu()
        accum_smape = ad_smape(gt, pred) * 0.5
        wmape = wmape_func(gt, pred)
        mae_val = F.l1_loss(label, predict)
        
        return mae_val, wmape, np.mean(adj_smape), accum_smape

def cal_error_metrics(gt, forecasts):
    # Absolute errors
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(forecasts):
        forecasts = forecasts.detach().cpu().numpy()

    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)
    

def print_error_metrics(y_test, y_hat, rescaled_y_test, rescaled_y_hat):

    if torch.is_tensor(y_test):
        y_test = y_test.detach().cpu().numpy()
    if torch.is_tensor(y_hat):
        y_hat = y_hat.detach().cpu().numpy()
    if torch.is_tensor(rescaled_y_test):
        rescaled_y_test = rescaled_y_test.detach().cpu().numpy()
    if torch.is_tensor(rescaled_y_hat):
        rescaled_y_hat = rescaled_y_hat.detach().cpu().numpy()

    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)

    sample_mae = np.mean(np.abs(y_test - y_hat), axis=1) 
    sample_mae_rescaled = np.mean(np.abs(rescaled_y_test - rescaled_y_hat), axis=1)

    mae_25 = np.percentile(sample_mae, 25)
    mae_75 = np.percentile(sample_mae, 75)

    mae_25_rescaled = np.percentile(sample_mae_rescaled, 25)
    mae_75_rescaled = np.percentile(sample_mae_rescaled, 75)


    ad_smape = SymmetricMeanAbsolutePercentageError()
    y_test_t = torch.tensor(y_test)
    y_hat_t = torch.tensor(y_hat)
    adj_smape_list = [ad_smape(y_test_t[i], y_hat_t[i]) * 0.5 for i in range(y_test_t.size(0))]
    adj_smape = np.mean([val.item() for val in adj_smape_list])


    print(f"MAE: {mae:.3f}  (25%: {mae_25:.3f}, 75%: {mae_75:.3f})")
    print(f"WAPE: {wape:.3f}")
    print(f"Rescaled MAE: {rescaled_mae:.3f}  (25%: {mae_25_rescaled:.3f}, 75%: {mae_75_rescaled:.3f})")
    print(f"Rescaled WAPE: {rescaled_wape:.3f}")
    print(f"Adj SMAPE: {adj_smape:.3f}")

    return mae, wape, rescaled_mae, rescaled_wape, mae_25, mae_75, mae_25_rescaled, mae_75_rescaled, adj_smape
