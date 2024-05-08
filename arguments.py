import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int, 
        default=23, 
        help='Random seed.',
    )

    parser.add_argument(
        '--data_path',
        type=str, 
        default='./data/MIND-large', 
        help='Path of data set.',
    )

    parser.add_argument(
        '--log_path',
        type=str,
        default='./log', 
        help='Path of log file.',
    )

    parser.add_argument(
        '--log_tag',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--npratio',
        type=int, 
        default=4, 
        help='Ratio of positive to negative samples is equal to <npratio>.',
    )

    parser.add_argument(
        '--ctitle_size',
        type=int,
        default=20,
        help='Pad or truncate the length of candidate news title to <ctitle_size>.',
    )

    parser.add_argument(
        '--htitle_size',
        type=int,
        default=10,
        help='Pad or truncate the length of historical news title to <htitle_size>.',
    )

    parser.add_argument(
        '--max_his_size',
        type=int,
        default=50,
        help='Maximum length of the history interaction. (truncate old if necessary).',
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='bert-base-uncased',
        help='Pretrained language model.',
    )

    parser.add_argument(
        '--prompt',
        type=str,
        default='<CANDIDATE> ? [MASK] , <USER>',
        help='A template that converts the input into a specific form.',
    )

    parser.add_argument(
        '--nsep_token',
        type=str,
        default='[NSEP]',
        help='A special token used to split news.',
    )

    parser.add_argument(
        '--prefix_size',
        type=int,
        default=1,
        help='Number of prefix tokens.',
    )

    parser.add_argument(
        '--generator_hidden_size',
        type=int,
        default=512,
        help='The hidden size of the prompt generator.',
    )

    parser.add_argument(
        '--cate_dim',
        type=int,
        default=32,
        help='Dimensions of news category embedding.',
    )

    parser.add_argument(
        '--gru_num_layers',
        type=int,
        default=1,
        help='Number of recurrent layers.',
    )

    parser.add_argument(
        '--tau',
        type=float,
        default=1.0,
        help='Non-negative scalar temperature.',
    )

    parser.add_argument(
        '--n_labels',
        type=int,
        default=2,
        help='Number of labels.',
    )

    parser.add_argument(
        '--val_ratio',
        type=float, 
        default=0.05, 
        help='Split <val_ratio> from training set as the validation set.',
    )

    parser.add_argument(
        '--search_epochs',
        type=int, 
        default=1,
    )

    parser.add_argument(
        '--train_epochs',
        type=int, 
        default=3,
    )

    parser.add_argument(
        '--train_batch_size',
        type=int, 
        default=16, 
        help='Batch size during training.',
    )

    parser.add_argument(
        '--infer_batch_size',
        type=int, 
        default=64,
        help='Batch size during inference.',
    )

    parser.add_argument(
        '--learning_rate_bert',
        type=float, 
        default=0.00001,
        help='Learning rate just for bert.',
    )

    parser.add_argument(
        '--learning_rate_prompt',
        type=float, 
        default=0.0003,
    )

    parser.add_argument(
        '--learning_rate_answer',
        type=float, 
        default=0.001,
    )

    parser.add_argument(
        '--weight_decay',
        type=float, 
        default=0.003,
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='./checkpoint', 
        help='Path of checkpoint.',
    )

    parser.add_argument(
        '--ckpt_name',
        type=str,
        default='model_checkpoint.pth',
    )

    parser.add_argument(
        '--ncols',
        type=int,
        default=80,
        help='Parameters of tqdm: the width of the entire output message.',
    )

    args = parser.parse_args()
    return args
