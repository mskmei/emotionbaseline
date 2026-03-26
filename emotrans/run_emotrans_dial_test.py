import os
import sys
import yaml
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from global_var import *

sys.path.append(utils_dir)

from config import config
from processor import Processor
from utils_processor import set_rng_seed


def main():
    # Original training style, but with dataset alias that has test.json replaced by DIAL.
    args = config(task="", dataset="meld_dial", framework=None, model="emotrans")

    with open("./configs/emotrans.yaml", "r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)
    args.train.update(run_config["train"])
    args.model.update(run_config["model"])

    args.train["inference"] = False
    args.train["do_valid"] = True
    args.train["do_test"] = True
    args.train["test_every_epoch"] = True
    args.train["log_step_rate"] = 1.0
    args.train["save_cls_report"] = True
    args.train["report_dir"] = "./saved/meld_dial_metrics"
    args.train["report_style"] = "anjs"
    args.train["report_label_map"] = {
        "anger": "A",
        "joy": "J",
        "neutral": "N",
        "sadness": "S",
        # Optional collapse for extra MELD labels if predicted.
        "surprise": "J",
        "fear": "S",
        "disgust": "A",
    }

    args.logger["display"].extend(["arch", "scale", "weight", "flow_num"])

    set_rng_seed(args.train["seed"])

    from Model_EmoTrans import import_model

    model, dataset = import_model(args)
    processor = Processor(args, model, dataset)
    result = processor._train()

    print("[Done] valid:", result["valid"])
    print("[Done] test (dial):", result["test"])
    print("[Done] reports at:", args.train["report_dir"])


if __name__ == "__main__":
    main()
