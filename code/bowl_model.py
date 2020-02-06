import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
from transformers.modeling_albert import AlbertConfig, AlbertModel
from transformers.modeling_xlnet import XLNetConfig, XLNetModel
from transformers.modeling_xlm import XLMConfig, XLMModel
from transformers.modeling_gpt2 import GPT2Model, GPT2Config, GPT2PreTrainedModel, Block
import bowl_db


class TransfomerModel(nn.Module):
    def __init__(self, cfg):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),            
        )        
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        #seq_length = self.cfg.seq_len
        #position_ids = torch.arange(seq_length, dtype=torch.long, device=cate_x.device)
        #position_ids = position_ids.unsqueeze(0).expand((batch_size, seq_length))
        #position_emb = self.position_emb(position_ids)
        #seq_emb = (seq_emb + position_emb)        
        #seq_emb = self.ln(seq_emb)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y


class DSB_BertModel(nn.Module):
    def __init__(self, cfg):
        super(DSB_BertModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertModel(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),            
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]        
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y


class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)        
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.encoder = nn.LSTM(cfg.hidden_size, 
                            cfg.hidden_size, cfg.nlayers, dropout=cfg.dropout, batch_first=True)           
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )        
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        _, (h, c) = self.encoder(seq_emb)
        sequence_output = h[-1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y


class DSB_GPT2Model(nn.Module):
    def __init__(self, cfg):
        super(DSB_GPT2Model, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)        
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        self.config = GPT2Config( 
            3, # not used
            n_positions=cfg.seq_len,
            n_ctx=cfg.hidden_size,
            n_embd=cfg.hidden_size,
            n_layer=cfg.nlayers,
            n_head=cfg.nheads,
            #embd_pdrop=cfg.dropout,
            #attn_pdrop=cfg.dropout,                 
        )
        self.encoder = GPT2Model(self.config)
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),            
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y


class DSB_ALBERTModel(nn.Module):
    def __init__(self, cfg):
        super(DSB_ALBERTModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)        
        def get_cont_emb():
            return nn.Sequential(                
                nn.Linear(cont_col_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size)                
            )
        self.cont_emb = get_cont_emb()
        self.config = AlbertConfig( 
            3, # not used
            embedding_size=cfg.emb_size*cate_col_size + cfg.hidden_size,
            hidden_size=cfg.emb_size*cate_col_size + cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            #num_hidden_groups=1,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,            
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
            max_position_embeddings=cfg.seq_len,
            type_vocab_size=1,
            #initializer_range=0.02,
            #layer_norm_eps=1e-12,
        )        
        
        self.encoder = AlbertModel(self.config)
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size + cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x)        
        cont_emb = self.cont_emb(cont_x)
        
        cate_emb = cate_emb.view(batch_size, self.cfg.seq_len, -1)        
        cont_emb = cont_emb.view(batch_size, self.cfg.seq_len, -1)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y


class DSB_XLNetModel(nn.Module):
    def __init__(self, cfg):
        super(DSB_XLNetModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        self.config = XLNetConfig( 
            3, # not used            
            d_model=cfg.hidden_size,
            n_layer=cfg.nlayers,
            n_head=cfg.nheads,
            d_inner=cfg.hidden_size,
            #ff_activation="gelu",
            #untie_r=True,
            #attn_type="bi",
            #initializer_range=0.02,
            #layer_norm_eps=1e-12,
            dropout=cfg.dropout,
            #mem_len=None,
            #reuse_len=None,
            #bi_data=False,
            #clamp_len=-1,
            #same_length=False,
            #summary_type="last",
            #summary_use_proj=True,
            #summary_activation="tanh",
            summary_last_dropout=cfg.dropout,
            #start_n_top=5,
            #end_n_top=5,
        )        
        
        self.encoder = XLNetModel(self.config)
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x)        
        cont_emb = self.cont_emb(cont_x)
        
        cate_emb = cate_emb.view(batch_size, self.cfg.seq_len, -1)        
        cont_emb = cont_emb.view(batch_size, self.cfg.seq_len, -1)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y

    
class DSB_XLMModel(nn.Module):
    def __init__(self, cfg):
        super(DSB_XLMModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        self.config = XLMConfig( 
            3, # not used
            emb_dim=cfg.hidden_size,
            n_layers=cfg.nlayers,
            n_heads=cfg.nheads,
            dropout=cfg.dropout,
            attention_dropout=cfg.dropout,
            gelu_activation=True,
            sinusoidal_embeddings=False,
            causal=False,
            asm=False,
            n_langs=1,
            use_lang_emb=True,
            max_position_embeddings=cfg.seq_len,
            embed_init_std=(cfg.hidden_size) ** -0.5,
            layer_norm_eps=1e-12,
            init_std=0.02,
            bos_index=0,
            eos_index=1,
            pad_index=2,
            unk_index=3,
            mask_index=5,
            is_encoder=True,
            summary_type="first",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=cfg.dropout,
            start_n_top=5,
            end_n_top=5,
            mask_token_id=0,
            lang_id=0,     
        )        
        
        self.encoder = XLMModel(self.config)
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),            
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
             
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y
    

encoders = {
    'LSTM':LSTMModel,
    'TRANSFORMER':TransfomerModel,
    'BERT':DSB_BertModel,    
    'GPT2':DSB_GPT2Model,
    'ALBERT':DSB_ALBERTModel,
    'XLNET':DSB_XLNetModel,
    'XLM':DSB_XLMModel,
    
}
