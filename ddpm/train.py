import argparse
import traceback
import logging
import sys
import os
import torch
import numpy as np

from runners.diffusion import Diffusion
from functions import get_config_and_setup_dirs

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--ckpt_folder", type=str, help="Path to folder with pretrained model. Only for forgetting training."
    )
    parser.add_argument(
        "--mode", type=str, default='train', help="['train', 'forget']"
    )
    parser.add_argument(
        "--label_to_forget", type=int, default=0, help="Class label 0-9 to forget. Only for forgetting training."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    # parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=2.0,
        help="classifier-free guidance conditional strength"
    )
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    config = get_config_and_setup_dirs(os.path.join("configs", args.config))

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    args, config = parse_args_and_config()
    if args.mode == 'train':
        logging.info("Writing log file to {}".format(config.log_dir))
        
    try:
        if args.mode == 'train':
            runner = Diffusion(args, config)
            runner.train()
        elif args.mode == 'forget':
            runner = Diffusion(args, config)
            runner.train_forget()
    except Exception:
        logging.error(traceback.format_exc())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
