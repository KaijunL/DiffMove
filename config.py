from dataclasses import dataclass, field


@dataclass
class GeolifConfig:
    # reproducing configuration
    seed: int = field(metadata={"help": "to reproducing the results in the paper."}, default=2021)

    dataset: str = field(metadata={"help": "string of dataset"}, default="foursquare")


    vocab_path: str = field(metadata={"help": "the path of vocab file"},
                            default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/pos.vocab.txt")
    dist_path: str = field(metadata={"help": "the path of distance file"},
                           default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/vocabs_dist.txt")
    train_path: str = field(metadata={"help": "the path of training data"},
                            default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/pos.train.txt")
    eval_path: str = field(metadata={"help": "the path of validation data"},
                           default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/pos.validate.txt")
    test_path: str = field(metadata={"help": "the path of testing data"},
                           default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/pos.test.txt")
    save_dir: str = field(metadata={"help": "the path for saving model"}, default="./save_models/")
    dump_emb_path: str = field(metadata={"help": "the path for saving embedding"},
                               default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/emb_w2i.pkl")

    embedding_table_path: str = field(metadata={"help": "the path for loading embedding_table"},
                                      default="../Dataset/v3_delta_0.005_window_size_6_mask_num_10/embedding_table_hidden_size35.emb")

    load_embed_table: bool = field(metadata={"help": "Pre-load well trained embedding table"}, default=False)


    # training configuration
    device: str = field(metadata={"help": "the running device"}, default="cuda")
    epochs: int = field(metadata={"help": "training epochs"}, default=800)
    batch_size: int = field(metadata={"help": "the training/validation/testing batch size"}, default=30) #original 16
    dropout_p: float = field(metadata={"help": "dropout rate"}, default=0.0)
    step: int = field(metadata={"help": "the steps of GGNN"}, default=2)
    lr: float = field(metadata={"help": "learning rate"}, default=1e-3)
    l2: float = field(metadata={"help": "weight decay"}, default=1e-5)
    patience: float = field(metadata={"help": "patience for early stopping"}, default=60)
    dist_loss: bool = field(metadata={"help": "add distance loss"}, default=False)  #original True
    alpha: float = field(metadata={"help": "loss balance weight"}, default=0.10)


    # model configuration
    hidden_size: int = field(metadata={"help": "hidden size of the model"}, default=128)  #original 128
    cross_n_heads: int = field(metadata={"help": "num of heads in cross attention layer"}, default=4)  #original 4 #actually cross_n_heads not used  here


    nfold: int = field(metadata={"help": "for 5fold test (valid value:[0-4])"}, default=0)
    diff_config_path: str = field(metadata={"help": "the path of diffusion config"}, default="base.yaml")
    unconditional: bool = field(default=False)
    testmissingratio: float = field(default=0.5)
    nsample: int= field(default=4)  #original 4


    use_historical_trajs: bool = field(default=False)

    GNN_construct_by_masked_current_trajs: bool = field(default=False)
    test_score_use_matching: bool = field(default=True)

    pos_encode: bool = field(default=False)

    use_side_info: str = field(metadata={"help": "side info use"}, default="history") #history, current, none
    No_side_info: bool = field(default=False)
    use_spatial_blk: bool = field(default=True)

    K: int = field(default=10)  # original 10
    theta: int = field(default=1000)  # original 1000
    shuffle: bool = field(default=False)