import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        # Intra- and Inter-modal Transformers
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)

        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))

        # Multimodal-level Gated Fusion
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # Emotion Classifier
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob
