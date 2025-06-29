from argparse import ArgumentParser

def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    
    #### FL args ####
    parser.add_argument('--no_fl', action='store_true', default=False, help="use centralized training instead of federated learning")
    parser.add_argument('--num_clients', type=int, default=50, help="number of clients")
    parser.add_argument('--test_size', type=float, default=0.166, help="test size")
    parser.add_argument('--rounds', type=int, default=10, help="number of communication rounds")
    parser.add_argument('--non_iid_alpha', type=float, default=None, help="non-iid alpha")
    parser.add_argument('--synthetic_count', type=int, default=0, help="number of synthetic samples to generate")
    parser.add_argument('--scale_syn', action='store_true', default=False, help="Use to scale synthethic sampling")
    
    
    #### Data args ####
    
    # -- Common data args
    parser.add_argument('--dataset', type=str, default='syntehetic', help="choose which data to use: synthetic or BrainTumor or MIMIC3")
    parser.add_argument('--dataset_path', type=str, default=None, help="path to preprocessed dataset. If None, use the default path.")
    parser.add_argument('--total_samples', type=int, default=None, help="total number of samples in the train set for CL")
    parser.add_argument('--overwrite_cache', action='store_true', default=False, help="overwrite cache")
    parser.add_argument('--result_dir', type=str, default='results', help="directory to save results")
    parser.add_argument('--samples_per_client', type=int, default=None, help="number of samples per client")
    parser.add_argument('--balancing', type=str, default=None, help="balancing method. Options: over, under, smote")
    parser.add_argument('--ir', type=float, default=None, help="imbalance ratio. must be above 1.0")
    
    # -- MIMIC-III data args
    parser.add_argument('--gcp_credentials', type=str, default=None, help="path to GCP credentials")
    parser.add_argument('--remove_outliers', action='store_true', default=False, help="Apply all outlier strategies")

    # -- BrainTumor data args
    parser.add_argument('--img_size', type=int, default=64, help="image size")
    
    
    #### Model args ####
    parser.add_argument('--local_ep', type=int, default=10, help="number of local epochs: E")
    parser.add_argument('--lr', type=float, default=0.001, help="client learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--batch_size', type=int, default=10, help="batch size")
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gradient_clipping', action='store_true', default=False, help="Use gradient clipping") 
    parser.add_argument('--early_stopping', action='store_true', default=False, help="Use early stopping")
    parser.add_argument('--early_stopping_patience', type=int, default=5, help="patience for early stopping")
    parser.add_argument('--early_stopping_metric', type=str, default='loss', help="metric for early stopping. Options: loss, accuracy, auc")
    parser.add_argument('--early_stopping_use_testset', action='store_true', default=False, help="use test set for early stopping")
    parser.add_argument('--early_stopping_disregard_rounds', type=int, default=0, help="number of rounds to disregard for early stopping")
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout rate")
    parser.add_argument('--threshold', type=float, default=0.5, help="inference threshold for classification")
    
    #### MISC args ####
    parser.add_argument('--tags', type=str, default=None, help="tags for the prefect run")
    parser.add_argument('--run_name', type=str, default='test', help="name of the run")
    parser.add_argument('--synthetic_balancing', action='store_true', default=False, help="balance synthetic data")
    return parser