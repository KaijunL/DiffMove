import numpy as np
import torch
import torch.nn as nn

from diff_models import diff_move, DiffusionEmbedding

import torch.nn.functional as F
#from .utils.nn import mean_flat
from torch.nn import Module, Parameter
import math

class base(nn.Module):
#    def __init__(self, target_dim, config, device):
    def __init__(self, target_dim, main_config,config,num_of_candidates,embedding_table, device, Amatrix, edge_weight):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]


     #   self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim  # original commented on Nov 16 2023
        self.emb_total_dim = self.emb_time_dim #+ self.emb_feature_dim

        if self.is_unconditional == False:

            if main_config.use_side_info == "history":
                if main_config.dataset == "foursquare":
                     self.emb_total_dim += 1+5
                if main_config.dataset == "geolife":
                   #  self.emb_total_dim += 1+3  # original commented on Nov 30 2023
                #     self.emb_total_dim += 1 + 1
                    self.emb_total_dim += 1+4  #  new geolife
            elif main_config.use_side_info == "current":  # B, 1, L, K
                self.emb_total_dim += 2

            else:
              #  self.emb_total_dim += 2  # for conditional mask with seq_hidden_current
                self.emb_total_dim += 1  # default setting no seq_hidden_current or seq_hidden_history for side info

        elif self.is_unconditional == True:
            if main_config.use_side_info == "history":
                if main_config.dataset == "foursquare":
                     self.emb_total_dim += 5
                if main_config.dataset == "geolife":
                   #  self.emb_total_dim += 1+3  # original commented on Nov 30 2023
                #     self.emb_total_dim += 1
                    self.emb_total_dim += 1+4  #  new geolife

            elif main_config.use_side_info == "current":  # B, 1, L, K
                self.emb_total_dim += 1

            else:
              #  self.emb_total_dim += 1  # for conditional mask with seq_hidden_current
                self.emb_total_dim += 0  # To run and test unconditional model with no side info (side info= none or nothing)

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config_diff["num_steps"],
            embedding_dim=config_diff["diffusion_embedding_dim"],
        )


        #self.diffmodel = diff_move(config_diff,  input_dim) #original
        self.diffmodel = diff_move(main_config,config_diff,num_of_candidates,embedding_table,input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)


        self.dropout = nn.Dropout(p=config["model"]["dropout_p"])
        self.config = config
        self.linear_transform = nn.Linear(self.target_dim,num_of_candidates) #, bias=True)
        self.ID_embedding = embedding_table
        with torch.no_grad(): #original commented on Nov 16 2023
            self.linear_transform.weight = embedding_table.weight #original commented on Nov 16 2023

        self.logits_mode = config["model"]["logits_mode"]



#         calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alpha)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alpha)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alpha)

        self.alphas_cumprod_prev = np.append(1.0, self.alpha[:-1])
        self.posterior_variance = (
                self.beta * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alpha)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.posterior_mean_coef1 = (
                self.beta * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(self.alpha_hat)
                / (1.0 - self.alpha)
        )

        self.loss_weight = torch.from_numpy(self.alpha / (1-self.alpha))
        self.loss_weight = self.loss_weight.to(self.device)

        self.Amatrix = Amatrix
        self.edge_weight = edge_weight

        self.main_config = main_config


    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask_original(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_randmask_my_diffu_move_working(self, current_trajs,observed_masks,missing_ratio):

        observed = current_trajs.masked_fill(current_trajs!=0,1).reshape(-1)#.copy()  #(bs*48)
        obs_indices = torch.where(observed)[0].tolist()  #(bs*48)
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        observed[miss_indices] = False
        #cond_mask = observed

        #targets = torch.masked_select(current_trajs, target_mask.bool())
        targets = torch.index_select(current_trajs.reshape(-1), 0, torch.LongTensor(miss_indices).to(current_trajs.device))

        #cond_mask = observed.reshape(current_trajs.shape)
        cond_mask = observed.reshape(-1,current_trajs.size(-1)) #bs,seqlen

        cond_mask = cond_mask.unsqueeze(1)  # (bs,1,48)
        cond_mask = cond_mask.repeat(1, observed_masks.size(1),1) # (bs,35,48)

        index = torch.LongTensor(miss_indices).to(current_trajs.device)

        indices_mask = torch.zeros_like(current_trajs).reshape(-1)
        indices_mask[miss_indices] = 1.0
        indices_mask = indices_mask.reshape(-1, current_trajs.size(-1))


        return cond_mask,targets, index, indices_mask

    def get_randmask(self, current_trajs,observed_masks,miss_indices):

        observed = current_trajs.masked_fill(current_trajs!=0,1).reshape(-1)#.copy()  #(bs*48)
        '''obs_indices = torch.where(observed)[0].tolist()  #(bs*48)
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )'''
        observed[miss_indices] = 0
        #cond_mask = observed

        #targets = torch.masked_select(current_trajs, target_mask.bool())
      #  targets = torch.index_select(current_trajs.reshape(-1), 0, torch.LongTensor(miss_indices).to(current_trajs.device))
        targets = torch.index_select(current_trajs.reshape(-1), 0, miss_indices)

        #cond_mask = observed.reshape(current_trajs.shape)
        cond_mask = observed.reshape(-1,current_trajs.size(-1)) #bs,seqlen

        cond_mask = cond_mask.unsqueeze(1)  # (bs,1,48)
        cond_mask = cond_mask.repeat(1, observed_masks.size(1),1) # (bs,35,48)

      #  index = torch.LongTensor(miss_indices).to(current_trajs.device)

        indices_mask = torch.zeros_like(current_trajs).reshape(-1)
        indices_mask[miss_indices] = 1.0
        indices_mask = indices_mask.reshape(-1, current_trajs.size(-1))


        return cond_mask,targets, miss_indices, indices_mask #,masked_current_trajs

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
   #     feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  #original commented on Nov 16 2023

   #     side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)  #original commented on Nov 16 2023
   #     side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)  #original commented on Nov 16 2023
        side_info = time_embed.permute(0, 3, 2, 1)  # (B,*,K,L)
        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info  # (B,channels+1,K,L)




    def calc_loss_valid(

        self, observed_data, Graph_input, cond_mask, observed_mask, side_info, current_traj_IDs, indices_mask, validate, inputs, targets):


        diff_loss_sum = 0
        decoder_nll_sum = 0
        tT_loss_sum = 0

        for t in range(self.num_steps):  # calculate loss for all t
            '''loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )'''

        #    diff_loss,decoder_nll,tT_loss = self.calc_loss(observed_data,Graph_input, cond_mask, observed_mask, side_info, current_traj_IDs, indices_mask, is_train,inputs,targets, set_t=t)  #original working on 11 Dec 2023
            diff_loss, decoder_nll, tT_loss = self.calc_loss(observed_data, Graph_input, cond_mask, observed_mask,
                                                             side_info, current_traj_IDs, indices_mask, validate,
                                                             inputs, targets, set_t=t)
            '''diff_loss_sum += diff_loss.detach()
            decoder_nll_sum += decoder_nll.detach()
            tT_loss_sum += tT_loss.detach()'''
            diff_loss_sum += diff_loss#.detach()
            decoder_nll_sum += decoder_nll#.detach()
            tT_loss_sum += tT_loss#.detach()

        return diff_loss_sum / self.num_steps,   decoder_nll_sum/ self.num_steps, tT_loss_sum/ self.num_steps

    def calc_loss_original(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

       # predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L) #original
        predicted = self.diffmodel(total_input, side_info, t, self.Amatrix,self.edge_weight)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1) #original working L2 loss

    #    loss = residual.sum() / (num_eval if num_eval > 0 else 1) # L1 loss

        return loss

    def get_logits(self, hidden_repr,embedding_table): # hidden_repr bsz, seqlen, K
        if self.logits_mode == 1:
            return self.linear_transform(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.linear_transform.weight ** 2).sum(-1).view(-1, 1)  # vocab

            #text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            text_emb_t = torch.transpose(text_emb.reshape(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen

            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.linear_transform.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        elif self.logits_mode == 3: # original matching
            #if self.config.use_historical_trajs ==True:
                #candidate_poi = embedding_table.weight[3:]
            #else:
            candidate_poi = embedding_table.weight#[1:] #now (vocab-1,K) (1409,35) #error for cuda assert, since vocab num is less when cal celoss, it is not correct. #original is embedding_table.weight 
            # scores: batch_size * mask_num, n
            scores = torch.matmul(hidden_repr, candidate_poi.transpose(1, 0)) #bs,seqlen,vocab-1=1409

            return scores
        else:
            raise NotImplementedError

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t #(bs, seqlen,hidden_sizeK)
        logits = get_logits(reshaped_x_t,self.ID_embedding)  # bsz, seqlen, vocab #logits_mode3 bsz, seqlen,vocab-1
        # print(logits.shape)

        if self.main_config.dataset == "foursquare":
            logits[:, :, 0:3] = -np.inf
        elif self.main_config.dataset == "geolife":
          #  logits[:, :, 0:8] = -np.inf
        #    logits[:, :, 0:4] = -np.inf
            logits[:, :, 0:3] = -np.inf
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0,reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape) #input_ids bs,seqlen #original

        #decoder_nll = loss_fct(F.softmax(logits.view(-1, logits.size(-1))), input_ids.view(-1)).view(input_ids.shape)

        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        # if mask != None:
        #     #decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1)
        #     decoder_nll = decoder_nll.mean()
        # else:
        #     #decoder_nll = decoder_nll.mean(dim=-1)
        #     decoder_nll = decoder_nll.mean()

      #  return decoder_nll
        return decoder_nll.mean()


    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )




    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alpha, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance #bs,seqlen,K


    def calc_loss(
        #self, observed_data, cond_mask, observed_mask, side_info,targets, is_train, set_t=-1):
        #self, observed_data, cond_mask, observed_mask, side_info, current_traj_IDs,indices_mask, is_train, set_t = -1): #original
    #    self, observed_data, Graph_input, cond_mask, observed_mask, side_info, current_traj_IDs,indices_mask, is_train, inputs,targets, set_t = -1): #original working before 11 Dec 2023
        self, observed_data, Graph_input, cond_mask, observed_mask, side_info, current_traj_IDs, indices_mask, validate, inputs, targets, set_t = -1):

        B, K, L = observed_data.shape
    #    if is_train != 1:  # for validation #original working before 11 Dec 2023
        if validate != 0:  # for validation

            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (BS,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        #predicted = self.diffmodel(total_input, side_info, t)  # original
    #    predicted = self.diffmodel(total_input, side_info, t, self.Amatrix,self.edge_weight)#.to(self.device)  # (B,K,L)
        predicted = self.diffmodel(total_input, side_info, t,Graph_input, self.Amatrix, self.edge_weight)


        target_mask = observed_mask - cond_mask
        if self.config["model"]["use_noise_predictor"] == True:
            residual = (noise - predicted) * target_mask #original
        else:
            residual = (observed_data - predicted) * target_mask

        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)


        #decoder_nll = self._token_discrete_loss(current_sample, self.get_logits, current_traj_IDs, mask = indices_mask)  # embedding regularization
        #x_start = self.ID_embedding(current_traj_IDs)

        x_start_mean = self.ID_embedding(current_traj_IDs)

        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        # print(std.shape, )
        # x_start_log_var = 2 * th.log(std)
        x_start = self._get_x_start(x_start_mean, std)

        predicted = predicted.permute(0, 2, 1)
        if self.config["model"]["use_current_sample_for_train_celoss"] == True and self.config["model"]["use_noise_predictor"] == False:

            decoder_nll = self._token_discrete_loss(predicted, self.get_logits, current_traj_IDs, mask=indices_mask) #original
           # current_sample = predicted
        else:
            decoder_nll = self._token_discrete_loss(x_start, self.get_logits, current_traj_IDs,mask=indices_mask)  # embedding regularization
           # current_sample = x_start



        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.num_steps - 1]).to(x_start.device)) #bs,seqlen,K
        tT_loss = mean_flat(out_mean ** 2)

        tT_loss = tT_loss.mean()

        if self.config["model"]["use_loss_weight"] == True:
            loss = loss * self.loss_weight#_extract_into_tensor(self.loss_weight, t, loss.shape)
            decoder_nll = decoder_nll * self.loss_weight#_extract_into_tensor(self.loss_weight, t, loss.shape)
            tT_loss = tT_loss * self.loss_weight#_extract_into_tensor(self.loss_weight, t, loss.shape)

        #return loss, decoder_nll, tT_loss #original
        return loss.mean(),decoder_nll.mean(), tT_loss.mean()




    def q_posterior(self, x_start, x_t, t):

        """
            Compute the mean and variance of the diffusion posterior:
                q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    """
    Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
    the initial x, x_0.

    :param model: the model, which takes a signal and a batch of timesteps
                  as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
             - 'mean': the model mean output.
             - 'variance': the model variance output.
             - 'log_variance': the log of 'variance'.
             - 'pred_xstart': the prediction for x_0.
    """
    def p_mean_variance(self, current_sample, x, side_info, t, Graph_input, clip_denoised = True):
        #preds = self.model_predictions(x, t, x_self_cond)
        #x_start = preds.pred_x_start
       # x_start = self.diffmodel(x, side_info, t).to(self.device) #original
        x_start = self.diffmodel(x, side_info, t, Graph_input, self.Amatrix,self.edge_weight).to(self.device)

        #x_start.to(self.device)
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = current_sample, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    '''@torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x,  side_info = x_self_cond, t = batched_times, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start'''

    '''@torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img'''


    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input
    

    def impute(self, observed_data, cond_mask, side_info, n_samples,Graph_input):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)

                
                else:

                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L) #original working


                if self.config["model"]["use_noise_predictor"] == True:
                    #predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device)) #original working 3 Oct 2023
                #    predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).repeat(B).to(self.device)) #original working Nov 11 2023
                    predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).repeat(B).to(self.device),Graph_input,self.Amatrix,self.edge_weight)  # modified Nov 11 2023
                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                ) ** 0.5
                        current_sample += sigma * noise

                else:

                   # model_mean, _, model_log_variance, x_start = self.p_mean_variance(current_sample, diff_input, side_info, torch.tensor([t]).to(self.device), clip_denoised=True) #original working 3 Oct 2023
                    model_mean, _, model_log_variance, x_start = self.p_mean_variance(current_sample, diff_input,
                                                                                      side_info,
                                                                                      torch.tensor([t]).repeat(B).to(self.device), Graph_input,
                                                                                      clip_denoised=True)
                    noise = torch.randn_like(current_sample) if t > 0 else 0.  # no noise if t == 0
                    predicted = model_mean + (0.5 * model_log_variance).exp() * noise

            if self.config["model"]["use_noise_predictor"] == True:
                imputed_samples[:, i] = current_sample.detach()  #commented original 7 Aug 2023
            else:
                imputed_samples[:, i] = predicted.detach()

        return imputed_samples

    def construct_graph(self, inputs):
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in range(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    #    return alias_inputs, A, items  #original

        return np.array(alias_inputs), np.array(A), np.array(items)

    def forward(self, input,targets, n_samples,embedding_table, is_train=1, validate=0): ##inputs, targets, anchor_points, pos_points, neg_points, pos_weights = batch[:-5], batch[-5], batch[-4], batch[-3],batch[-2], batch[-1]
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
 #           _,
            cut_length,

            Graph_input

        ) = self.process_data(input)

    #    targets = cut_length #temporary storage of targets as a LongTensor
        targets = targets

     #    #if self.main_config.t_in_GNN == False:
     #
     #    historical_inputs = input[3]
     #    historical_input_A = input[2]
     #    historical_alias_inputs = input[1]
     #
     #    # Historical trajectory GNN
     #    historical_sessions = self.ID_embedding(historical_inputs)
     #    historical_sessions = self.GNN(historical_sessions, historical_input_A)
     #    get_history = lambda i: historical_sessions[i][historical_alias_inputs[i]]
     #    # batch_size * history_length, seq_length, hidden_size
     #    seq_hidden_history = torch.stack([get_history(i) for i in torch.arange(len(historical_alias_inputs)).long()])
     #    # batch_size * history_length, seq_length, hidden_size
     #    seq_hidden_history = self.pos_encoding_layer(seq_hidden_history)
     #    # batch_size * history_size, seq_length, hidden_size -> batch_size, history_size, seq_length, hidden_size
     # #   seq_hidden_history = seq_hidden_history.view(-1, self.history_size, seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])
     #   # seq_hidden_history = seq_hidden_history.view(seq_hidden_history.shape[0],-1,seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])  #B, history_size, L, K
     #    seq_hidden_history = seq_hidden_history.view(self.main_config.batch_size,-1,seq_hidden_history.shape[-2],seq_hidden_history.shape[-1])  #B, history_size, L, K
        
        
        
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
#            cond_mask = self.get_randmask(observed_mask)
            #cond_mask,targets,index, indices_mask = self.get_randmask(input[4],observed_mask,missing_ratio=0.2)
       #     cond_mask,targets,index, indices_mask = self.get_randmask(input[-2],observed_mask,missing_ratio=0.2)    #original commented 17 Nov 2023
          #  cond_mask = gt_mask
            cond_mask, targets, index, indices_mask = self.get_randmask(input[-2], observed_mask,input[-7])



    #    loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss_func = self.calc_loss if validate == 0 else self.calc_loss_valid

        #temp_selection = input[5].cpu().numpy()
        #selected = indices_mask[temp_selection]
        #selected[:,:] =1.0
        #indices_mask = indices_mask.reshape(-1,input[4].size(-1))

        if is_train == 0:  #original commented 17 Nov 2023

            '''indices_mask = torch.zeros_like(input[4]).float().reshape(-1)
            cur_mask_pos = input[5]'''
            indices_mask = torch.zeros_like(input[-2]).float().reshape(-1)
            cur_mask_pos = input[-1]
            # cur_mask_pos: batch_size, mask_num -> index: batch_size * mask_num
            temp = torch.arange(cur_mask_pos.shape[0]).to(cur_mask_pos.device).view(-1, 1)
            #index = (cur_mask_pos + temp * input[4].shape[1]).view(-1)
            index = (cur_mask_pos + temp * input[-2].shape[1]).view(-1)
            indices_mask[index] = 1.0
            #indices_mask = indices_mask.reshape(-1, input[4].size(-1))
            indices_mask = indices_mask.reshape(-1, input[-2].size(-1))



        side_info = self.get_side_info(observed_tp, cond_mask)  # original
        diff_loss, discrete_celoss, tT_loss = loss_func(observed_data, Graph_input, cond_mask, observed_mask, side_info,
                                                        input[-2], indices_mask, validate, input, targets)

        return diff_loss, discrete_celoss, tT_loss




#    def evaluate(self, batch, n_samples):
    def evaluate(self, input, n_samples,embedding_table,config):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,

            Graph_input

        ) = self.process_data(input)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask) #original
        #    side_info = self.get_side_info(observed_tp, cond_mask, seq_hidden_history, seq_hidden_current)
            #    side_info = self.get_side_info(observed_data, cond_mask)

            #samples = self.impute(observed_data, cond_mask, side_info, n_samples) #original
            samples = self.impute(observed_data, cond_mask, side_info, n_samples,Graph_input)

    #        for i in range(len(cut_length)):  # to avoid double evaluation
    #            target_mask[i, ..., 0 : cut_length[i].item()] = 0

            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
        #    samples_median = samples.median(dim=1)
       #     samples_median_values = samples_median.values
            samples_median_values = samples.mean(dim=1)  #not yet change the name of "samples_median_values" for convienent use below later, should change to "mean" later

            cur_mask_pos = input[-1]
            # cur_mask_pos: batch_size, mask_num -> index: batch_size * mask_num
            temp = torch.arange(cur_mask_pos.shape[0]).to(cur_mask_pos.device).view(-1, 1)
            index = (cur_mask_pos + temp * samples_median_values.shape[1]).view(-1)
            #index = (cur_mask_pos + torch.arange(cur_mask_pos.shape[0]).to(cur_mask_pos.device).view(-1, 1) *
            #         samples_median_values.shape[1]).view(-1)
            # hybrid_embedding_: batch_size, seq_length, Feature_size_of_embedding_table -> batch_size * seq_length, Feature_size_of_embedding_table
            # batch_size * mask_num, Feature_size_of_embedding_table
            samples_embedding = samples_median_values.view(-1, self.target_dim).index_select(0, index)


            if config.test_score_use_matching == False:
                samples_embedding = samples_embedding.unsqueeze(0) # bs=10,index size = bs* num_of_targets_per_batch(10) = 10*10 =(100,), #cur_mask_pos (10,10) #samples_embedding (1,100,64)
                score = self.get_logits(samples_embedding,self.ID_embedding) #score (1,bs* num_of_targets_per_batch(10),num_of_candidates = max(w2i_dict.values()) + 1) = (1,100,1412)

                if config.dataset == "foursquare":
                    score[:, :, 0:3] = -np.inf
                elif config.dataset == "geolife":
                  #  score[:, :, 0:8] = -np.inf
                #    score[:, :, 0:4] = -np.inf
                    score[:, :, 0:3] = -np.inf
                '''score_mask = torch.ones_like(score)#.reshape(-1)
                if config.dataset == "foursquare":
                    score_mask[:,:,0:3] = 0
                elif config.dataset == "geolife":
                    score_mask[:, :, 0:8] = 0
                score = score * score_mask
                score = score.masked_fill(score==0,-np.inf)'''

                score = score.reshape(-1,score.size(2))
                score = F.log_softmax(score, dim=-1)  #check whether it can be commented
            else:
                if config.dataset == "foursquare":

                    # if config.use_historical_trajs == True:
                    #     candidate_poi = embedding_table.weight[3:]
                    # else:
                    #     candidate_poi = embedding_table.weight[1:]
                    candidate_poi = embedding_table.weight[3:]

                elif config.dataset == "geolife":

                #    candidate_poi = embedding_table.weight[8:]
             #       candidate_poi = embedding_table.weight[4:]
                    candidate_poi = embedding_table.weight[3:]
                scores = torch.matmul(samples_embedding, candidate_poi.transpose(1, 0))
                score = F.log_softmax(scores, dim=-1)

        #        return samples, observed_data, target_mask, observed_mask, observed_tp
        return score


class DiffMove(base):
    def __init__(self, main_config, config, num_of_candidates, embedding_table, device, target_dim,Amatrix, edge_weight):

        super(DiffMove, self).__init__( target_dim,main_config,config, num_of_candidates, embedding_table, device, Amatrix, edge_weight)

    def process_data(self, batch):

        observed_data = batch[-6].to(self.device).float()
        observed_mask = batch[-5].to(self.device).float()
        observed_tp = batch[-4].to(self.device).float()
        gt_mask = batch[-3].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        historical_inputs = batch[3]
        historical_input_A = batch[2]
        historical_alias_inputs = batch[1]
        masked_current_items = batch[6]
        masked_current_A = batch[5]
        masked_current_alias_inputs = batch[4]

        Graph_input = [historical_inputs,
            historical_input_A,
            historical_alias_inputs,
            masked_current_items,
            masked_current_A,
            masked_current_alias_inputs]

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,

            Graph_input

        )

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

'''def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))'''


