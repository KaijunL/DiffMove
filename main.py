import datetime
import os
import argparse
import datetime
import json
import yaml
import os

from main_model import DiffMove

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import HfArgumentParser, set_seed
from config import GeolifConfig
from utils import *  #modified  10 Dec 2023

from data_provider import TrajDataloder
from train import train
from test import test
import pickle
from pathlib import Path

import math
import numpy as np

if __name__ == "__main__":
    print(torch.__version__)

    parser = HfArgumentParser(GeolifConfig)
    config = parser.parse_args_into_dataclasses()[0]
    print(config)

    path = "diff_config/" + config.diff_config_path
    with open(path, "r") as f:
        diff_config = yaml.safe_load(f)

    diff_config["model"]["is_unconditional"] = config.unconditional
    diff_config["model"]["test_missing_ratio"] = config.testmissingratio



    print(json.dumps(diff_config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/Diffu_move" + str(config.nfold) + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)


    set_seed(config.seed)

    config.save_path = Path(config.save_dir) / "dataset_{}_hiddensize_{}_nheads_{}_distloss_{}_lr_{:.4f}_nsamp_{}_miss_{:.2f}_nopos_beta{:.2f}.chkpt".format(config.dataset, config.hidden_size, config.cross_n_heads, config.dist_loss, config.lr, config.nsample,config.testmissingratio,diff_config["diffusion"]["beta_end"])
    
    w2i_dict = load_w2ifile(config.vocab_path,config.use_historical_trajs,config.dataset)

    dist_dict = load_dist_dict(config.dist_path,config.dataset)
    dist_index_dict = make_dist_index_dict(w2i_dict, dist_dict)

    train_data_dict = preprocess(config.train_path, w2i_dict, config.dataset)
    valid_data_dict = preprocess(config.eval_path, w2i_dict, config.dataset)
    test_data_dict = preprocess(config.test_path, w2i_dict, config.dataset)

    device = torch.device(config.device)

    num_of_candidates = max(w2i_dict.values()) + 1


    if config.load_embed_table == False:

        embedding_table = nn.Embedding(max(w2i_dict.values()) + 1, config.hidden_size).to(device)
        stdv = 1.0 / math.sqrt(config.hidden_size)

        embedding_table.weight.data.normal_(-stdv, stdv)

        idx_ar = np.array(list(dist_index_dict.keys()))
        out_shp = idx_ar.max(0) + 1
        data = np.array(list(dist_index_dict.values()))
        M = np.zeros(shape=out_shp, dtype=data.dtype)
        M[tuple(idx_ar.T)] = data
        adj = get_similarity_adj(M, thr=0.1, force_symmetric=False, sparse=False)


        edge_weight = adj
        adj = M
        Amatrix = torch.tensor(adj, dtype=torch.float).to(device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)



    else:
        with open(config.embedding_table_path, "rb") as f:
            dataset_embedding = pickle.load(f)
            embedding_weight = torch.tensor(dataset_embedding[0], dtype=torch.float)
            embedding_table = nn.Embedding.from_pretrained(embedding_weight,freeze=True).to(device)

    n_samples = config.nsample


    train_dataloader = TrajDataloder(train_data_dict, dist_index_dict, device, w2i_dict, embedding_table, config,is_train=0, gen_pairs=config.dist_loss,K=config.K,theta=config.theta)
    valid_dataloader = TrajDataloder(valid_data_dict, dist_index_dict, device, w2i_dict, embedding_table, config, gen_pairs=False,K=config.K,theta=config.theta)
    test_dataloader = TrajDataloder(test_data_dict, dist_index_dict, device, w2i_dict, embedding_table, config, gen_pairs=False,K=config.K,theta=config.theta)


    model = DiffMove(config,diff_config, num_of_candidates, embedding_table, device, config.hidden_size, Amatrix,
                        edge_weight).to(device)

    # loss function

    criterion = nn.NLLLoss(ignore_index=0).cuda()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.l2)

    # train

    train(
        model,
        config,
        train_dataloader,
        embedding_table,
        dist_index_dict,
        n_samples,
        criterion,

        diff_config,
        test_loader = test_dataloader,
        valid_loader=valid_dataloader,
        foldername=foldername,

    )

    config.save_path = "./save_models/dataset_foursquare_hiddensize_128_nheads_4_distloss_True_lr_0.0010_nsamp_4_miss_0.50_nopos.chkpt"

  #  model.load_state_dict(torch.load(config.save_path))  #Newly commented for testing


    dump_emb_path = Path(
        config.save_dir) / "dataset_{}_hiddensize_{}_nheads_{}_distloss_{}_dropout_{}_alpha_{}_lr_{:.4f}_nopos_beta{:.2f}.emb".format(
        config.dataset, config.hidden_size, config.cross_n_heads, config.dist_loss, config.dropout_p, config.alpha,
        config.lr,diff_config["diffusion"]["beta_end"])

    embedding_matrix = embedding_table.weight.detach().cpu().numpy()
    pickle.dump([embedding_matrix, w2i_dict], Path(dump_emb_path).open("wb"))

    
