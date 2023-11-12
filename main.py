import argparse
import logging
import pathlib
import time

import gin

from bounce.bounce import Bounce
from bounce.util.data_loading import (
    download_uci_data,
    download_maxsat60_data,
    download_maxsat125_data,
)
from bounce.util.printing import BColors, BOUNCE_NAME


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=f"{BColors.LIGHTGREY} %(levelname)s:%(asctime)s - (%(filename)s:%(lineno)d) - %(message)s {BColors.ENDC}",
    )

    logging.info(BOUNCE_NAME)

    if not pathlib.Path("data/slice_localization_data.csv").exists():
        download_uci_data()

    if not pathlib.Path("data/maxsat/frb10-6-4.wcnf").exists():
        download_maxsat60_data()

    if not pathlib.Path(
        "data/maxsat/cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"
    ).exists():
        download_maxsat125_data()

    then = time.time()
    parser = argparse.ArgumentParser(
        prog=BOUNCE_NAME,
        description="Bounce: a Reliable Bayesian Optimization Algorithm for Combinatorial and Mixed Spaces",
        epilog="For more information, please contact the author.",
    )

    parser.add_argument(
        "--gin-files",
        type=str,
        nargs="+",
        default=["configs/default.gin"],
        help="Path to the config file",
    )
    parser.add_argument(
        "--gin-bindings",
        type=str,
        nargs="+",
        default=[],
    )

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    bounce = Bounce()
    bounce.run()

    gin.clear_config()
    now = time.time()
    logging.info(f"Total time: {now - then:.2f} seconds")
