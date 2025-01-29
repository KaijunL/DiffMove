import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_move(nn.Module):
    #def __init__(self, config, inputdim=2): #original
    def __init__(self, main_config,config, num_of_candidates, embedding_table, inputdim=2):

        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        #self.input_projection2= Conv1d_with_init(3*3, self.channels, 1) #for geolife history_size=3, train_dataloader.history_length =5, bs=3
        if main_config.dataset =="foursquare":
            self.input_projection2 = Conv1d_with_init(main_config.batch_size * 5, self.channels, 1)  #for foursquare history_size=5, train_dataloader.history_length =5, bs=
            self.input_projection3 = Conv1d_with_init(2+5, self.channels, 1)  #for foursquare history_size=5, train_dataloader.history_length =5, bs=
        if main_config.dataset == "geolife":
            self.input_projection2 = Conv1d_with_init(main_config.batch_size * 3, self.channels, 1) #for geolife history_size=3, train_dataloader.history_length =5, bs=
            self.input_projection3 = Conv1d_with_init(2+3, self.channels, 1) #for geolife history_size=3, train_dataloader.history_length =5, bs=

        self.main_config = main_config



        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    num_of_candidates = num_of_candidates,

                    embedding_table = embedding_table,
                    main_config = main_config,
                    config = config,
                )
                for _ in range(config["layers"])
            ]
        )

    #def forward(self, x, cond_info, diffusion_step): #original
    def forward(self, x, cond_info, diffusion_step,Graph_input, Amatrix,edge_weight):

        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)


        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            #x, skip_connection = layer(x, cond_info, diffusion_emb) #original

            x, skip_connection = layer(x, cond_info, diffusion_emb,Graph_input,Amatrix,edge_weight)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,num_of_candidates,embedding_table,main_config,config):
        super().__init__()

        self.channels = config["channels"]

        if main_config.dataset =="foursquare":
            self.input_projection5 = Conv1d_with_init(5, self.channels, 1)  #for foursquare history_size=5, train_dataloader.history_length =5, bs=
        if main_config.dataset == "geolife":
         #   self.input_projection5 = Conv1d_with_init(3, self.channels, 1) #for geolife history_size=3, train_dataloader.history_length =5, bs=
        #original
#            self.input_projection5 = Conv1d_with_init(1, self.channels,1)  # for geolife history_size=3, train_dataloader.history_length =5, bs=
            self.input_projection5 = Conv1d_with_init(4, self.channels, 1)

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels) #original working 4 Oct 2023
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.history_layer = get_torch_trans(heads=nheads, layers=1, channels=channels) #original layers=1

       # self.GNN_encoder = GatedGraphNN(channels, 2,channels,GNN_dropout=0.2)
 #       self.GNN_encoder = GCN(channels, channels, num_of_candidates, dropout=0.2)#.to(self.device)

        self.ID_embedding = embedding_table
        self.pos_encoding_layer = PosEncoder(48, main_config.hidden_size)

    #    self.pos_embed_layer = PosEmbedding(48, main_config.hidden_size)

        self.GNN = SessionGraph(main_config, num_of_candidates,diffusion_embedding_dim)
        self.main_config = main_config
        self.cross_attn_layer = PeriodMigrationCrossMultiHeadAttention(main_config.hidden_size,
                                                                       main_config.cross_n_heads)
        self.linear_one = nn.Linear(main_config.hidden_size, main_config.hidden_size, bias=True)
        self.linear_two = nn.Linear(main_config.hidden_size, main_config.hidden_size, bias=True)
        self.linear_three = nn.Linear(main_config.hidden_size, 1, bias=False)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward_history(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
       # y = self.history_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = self.history_layer(y)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward_trajectory(self, y):
        #B, channel, K, L = base_shape
        #if K == 1:
        #    return y
        #y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
       # y = self.history_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = self.history_layer(y)
        #y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    #def forward(self, x, cond_info, diffusion_emb):  #original
    def forward(self, x, cond_info, diffusion_emb,Graph_input, Amatrix,edge_weight):  #diffusion_emb (B,diffusion_embedding_dim)

        historical_inputs = Graph_input[0]
        historical_input_A = Graph_input[1]
        historical_alias_inputs = Graph_input[2]
        masked_current_items = Graph_input[3]
        masked_current_A = Graph_input[4]
        masked_current_alias_inputs = Graph_input[5]

        # Historical trajectory GNN
        historical_sessions = self.ID_embedding(historical_inputs)
        historical_sessions = self.GNN(historical_sessions, historical_input_A,diffusion_emb)
        get_history = lambda i: historical_sessions[i][historical_alias_inputs[i]]
        # batch_size * history_length, seq_length, hidden_size
        seq_hidden_history = torch.stack([get_history(i) for i in torch.arange(len(historical_alias_inputs)).long()])
        # batch_size * history_length, seq_length, hidden_size
        seq_hidden_history = self.pos_encoding_layer(seq_hidden_history)
   #     seq_hidden_history = self.pos_embed_layer(seq_hidden_history)

    #    seq_hidden_history = self.forward_trajectory(seq_hidden_history)

        # batch_size * history_size, seq_length, hidden_size -> batch_size, history_size, seq_length, hidden_size
        #   seq_hidden_history = seq_hidden_history.view(-1, self.history_size, seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])
        # seq_hidden_history = seq_hidden_history.view(seq_hidden_history.shape[0],-1,seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])  #B, history_size, L, K
        seq_hidden_history = seq_hidden_history.view(self.main_config.batch_size,-1,seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])  #B, history_size, L, K

        current_sessions = self.ID_embedding(masked_current_items)
        current_sessions = self.GNN(current_sessions, masked_current_A,diffusion_emb)
        get_current = lambda i: current_sessions[i][masked_current_alias_inputs[i]]
        # batch_size, seq_length, hidden_size
        seq_hidden_current = torch.stack(
            [get_current(i) for i in torch.arange(len(masked_current_alias_inputs)).long()])
        # batch_size, seq_length, hidden_size
        seq_hidden_current = self.pos_encoding_layer(seq_hidden_current)
  #      seq_hidden_current = self.pos_embed_layer(seq_hidden_current)

    #    seq_hidden_current = self.forward_trajectory(seq_hidden_current)

        # Period migration Cross Attention
        # batch_size, history_size, seq_length, hidden_size

        # seq_hidden_history = self.cross_attn_layer(seq_hidden_current, seq_hidden_history)
        # # B, history_size, L, K for seq_hidden_history  (3,3,48,64)
        # B, history_size, L, K = seq_hidden_history.shape
        # # x = x.reshape(B, -1, K * L)
        # seq_hidden_history = seq_hidden_history.permute(0, 1, 3, 2).reshape(B, history_size, K * L)
        # # x = torch.cat([x,seq_hidden_history], dim=1)
        # seq_hidden_history2 = self.input_projection5(seq_hidden_history) # (B,channel,K*L)


        seq_hidden_history2 = self.cross_attn_layer(seq_hidden_current, seq_hidden_history)
        # B, history_size, L, K for seq_hidden_history  (3,3,48,64)
        B, history_size, L, K = seq_hidden_history2.shape
        # x = x.reshape(B, -1, K * L)
        seq_hidden_history2 = seq_hidden_history2.permute(0, 1, 3, 2).reshape(B, history_size, K * L)
        # x = torch.cat([x,seq_hidden_history], dim=1)
        seq_hidden_history2 = self.input_projection5(seq_hidden_history2) # (B,channel,K*L)

        seq_hidden_history2 = F.relu(seq_hidden_history2) # (B,channel,K*L)

        base_shape = x.shape
        x = x.reshape(B, self.channels, K * L)

        if self.main_config.use_spatial_blk == True:
            x = x + seq_hidden_history2

        B, channel, _ = x.shape # (B,channel,K*L)
        # x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        concat_history_first = True
        concat_option = False
        if concat_option == False:
            y = self.forward_time(y, base_shape) # (B,channel,K*L)

            if concat_history_first == False:

                y = y + seq_hidden_history2

        #    y = self.forward_feature(y, base_shape)  # (B,channel,K*L)

            y = self.mid_projection(y)  # (B,2*channel,K*L) #original
        elif concat_option == True:
            y_1 = self.forward_time(y, base_shape)
            y = torch.cat([y_1, seq_hidden_history2.repeat(y_1.size(0),1,1)], dim=1)

     #   y= torch.cat([y_1, seq_hidden_history], dim=1)

      #  y = self.forward_time(y, base_shape)
      #  y = self.forward_feature(y, base_shape)  # (B,channel,K*L)

        '''y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2)#.reshape(B *L*channel, K)#.reshape(B * L, channel, K)
        #y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)


        y = self.GNN_encoder(y, Amatrix,edge_weight)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)'''
      #  y_1 = self.forward_time(y, base_shape)
      #  y_2 = self.forward_feature(y, base_shape)  # (B,channel,K*L)

      #  y = y_1 + y_2
    #    y = torch.cat([y_1,y_2],dim=1)

        #y = self.mid_projection(y)  # (B,2*channel,K*L)  #original


        _, cond_dim, _, _ = cond_info.shape # (B,cond_dim,K,L)

        seq_hidden_current = seq_hidden_current.unsqueeze(3).permute(0, 3, 2, 1)  # (B,1,K,L)
        seq_hidden_history = seq_hidden_history.reshape(B, history_size, K, L)#.permute(0, 1, 3, 2)  # (B,history_size,L,K)

        if self.main_config.use_side_info == "history":
            cond_info = torch.cat([cond_info, seq_hidden_history],dim=1)  # seq_hidden_history: batch_size, history_size, seq_length, hidden_size
        elif self.main_config.use_side_info == "current": #B, 1, K, L
            cond_info = torch.cat([cond_info, seq_hidden_current], dim=1)
        # else:
        #     cond_info = cond_info.reshape(B, cond_dim, K * L) # (B,cond_dim=channel+1=128+1,K*L)

        cond_info = cond_info.reshape(B, -1, K * L)

        reshaped = cond_info.reshape(B, -1, K, L)  # (B,128+1+5,K,L)
        shape = reshaped.shape
        #cond_info = self.forward_history(cond_info, shape)  # (B,cond_dim,K*L)

        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)

        if self.main_config.No_side_info == False:

            y = y + cond_info  # (B,2*channel,K*L)

        gate, filter = torch.chunk(y, 2, dim=1) # (B,channel,K*L)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) # (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1) # (B,channel,K*L)
        x = x.reshape(base_shape) # (B,channel,K,L)
        residual = residual.reshape(base_shape) # (B,channel,K,L)
        skip = skip.reshape(base_shape) # (B,channel,K,L)
        return (x + residual) / math.sqrt(2.0), skip



class PosEncoder(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / hidden_size) if i % 2 == 0 else -10000 ** ((1 - i) / hidden_size) for i in range(hidden_size)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(hidden_size)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(hidden_size, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x, transpose=True):
        if not transpose:
            x = x + self.pos_encoding
            return x
        return x + self.pos_encoding.transpose(0,1)


class PosEmbedding(nn.Module):
    def __init__(self, length, hidden_size):
        super().__init__()
        self.pos_embedding = nn.Embedding(length, hidden_size)

    def forward(self, x, transpose=None):
        if transpose is not None:
            raise ValueError("For now, transpose is not supported by PosEmbedding.")
        return x + self.pos_embedding.weight


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1,diffusion_embedding_dim=128):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2 + diffusion_embedding_dim
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden,diffusion_emb): #diffusion_emb: (B,diffusion_embedding_dim)
        edge_in_hidden = self.linear_edge_in(hidden)
        edge_out_hidden = self.linear_edge_out(hidden)
        input_in = torch.matmul(A[:, :, :A.shape[1]], edge_in_hidden) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], edge_out_hidden) + self.b_oah
    #    inputs = torch.cat([input_in, input_out], 2)  #inputs: (B*history_size,-1,2*hidden_size)

        diffusion_emb = diffusion_emb.unsqueeze(1).repeat(A.shape[0]//diffusion_emb.shape[0], A.shape[1], 1)
        inputs = torch.cat([input_in, input_out,diffusion_emb], 2)

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden,diffusion_emb):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden,diffusion_emb)
        return hidden


class SessionGraph(nn.Module):
    def __init__(self, opt, n_node, diffusion_embedding_dim):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hidden_size
        self.n_node = n_node
        self.gnn = GNN(self.hidden_size, opt.step,diffusion_embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hidden, A, diffusion_emb):
        hidden = self.gnn(A, hidden, diffusion_emb)
        return hidden

class PeriodMigrationCrossMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.linear_his_one = nn.Linear(hidden_size, hidden_size * n_heads, bias=True)
        self.linear_his_two = nn.Linear(hidden_size, hidden_size * n_heads, bias=True)
        self.compress_layer = nn.Linear(hidden_size * n_heads, hidden_size)

    def forward(self, seq_hidden_current, seq_hidden_history):
        # seq_hidden_history: batch_size, history_size, seq_length, hidden_size
        # seq_hidden_current: batch_size, seq_length, hidden_size

        batch_size = seq_hidden_history.shape[0]

        # batch_size, seq_length, hidden_size*n_heads
        his_q1 = self.linear_his_one(seq_hidden_current)
        # batch_size, history_size, seq_length, hidden_size*n_heads
        his_q2 = self.linear_his_two(seq_hidden_history)

        # batch_size*n_heads, seq_len, hidden_size
        his_q1 = torch.cat(his_q1.split(self.hidden_size, dim=-1), dim=0)
        # batch_size*n_heads, history_size, seq_len, hidden_size
        his_q2 = torch.cat(his_q2.split(self.hidden_size, dim=-1), dim=0)
        # batch_size*n_heads, hidden_size, history_size*seq_len
        his_q2_reshaped = his_q2.view(his_q2.shape[0], -1, his_q2.shape[-1]).transpose(1, 2)
        # batch_size*n_heads, seq_len, history_size, seq_len
        att_weights = torch.softmax(
            torch.bmm(his_q1, his_q2_reshaped).view(his_q1.shape[0], his_q1.shape[1], -1, his_q1.shape[1]), -1)
        # batch_size*n_heads, seq_len, history_size, hidden_size
        seq_hidden_history_attn = (att_weights.unsqueeze(-1) * his_q2.unsqueeze(1)).sum(-2).transpose(1, 2)
        # batch_size, seq_len, history_size, hidden_size*n_heads
        seq_hidden_history_attn = torch.cat(seq_hidden_history_attn.split(batch_size, 0), -1)

        # residual connection
        seq_hidden_history = self.compress_layer(seq_hidden_history_attn) + seq_hidden_history

        return seq_hidden_history



