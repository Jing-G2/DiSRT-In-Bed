import os

# from core.trainer import Trainer
from core.bodymap_trainer import Trainer
from utils.script_utils import create_argparser, args_dict_get_train_val_files

import warnings

warnings.filterwarnings("ignore")


def main():
    args_file = "./config/bodymap.json"
    parser = create_argparser(args_file)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict = args_dict_get_train_val_files(args_dict)

    print("Mode: " + args_dict["mode"])
    trainer = Trainer(args_dict)

    if args.mode in ["train", "finetune"]:
        trainer.train_model()
    else:
        trainer.test_model()

    print("Done")


if __name__ == "__main__":
    main()
