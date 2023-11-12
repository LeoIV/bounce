import logging
import math
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def download_uci_data():
    if not pathlib.Path("data/slice_localization_data.csv").exists():
        logging.info("slice_localization_data.csv not found. Downloading...")
        import urllib.request

        url = "https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip"
        logging.info(f"Downloading {url}")
        urllib.request.urlretrieve(url, "data/slice_localization_data.zip")
        logging.info("Download completed.")
        import zipfile

        with zipfile.ZipFile("data/slice_localization_data.zip", "r") as zip_ref:
            zip_ref.extractall("data")
        # delete .zip file
        pathlib.Path("data/slice_localization_data.zip").unlink()
        logging.info("Data extracted!")


def download_maxsat60_data():
    if not pathlib.Path("data/maxsat/frb10-6-4.wcnf").exists():
        logging.info("frb10-6-4.wcnf not found. Downloading...")
        import urllib.request

        url = "http://www.maxsat.udl.cat/11/benchs/wms_crafted.tgz"
        logging.info(f"Downloading {url}")
        urllib.request.urlretrieve(url, "data/maxsat/wms_crafted.tgz")

        import tarfile

        with tarfile.open("data/maxsat/wms_crafted.tgz", "r:gz") as tar:
            tar.extractall("data/maxsat")
            # move data/maxsat/wms_crafted/frb/frb10-6-4.wcnf to data/maxsat/frb10-6-4.wcnf
            pathlib.Path("data/maxsat/wms_crafted/frb/frb10-6-4.wcnf").rename(
                "data/maxsat/frb10-6-4.wcnf"
            )
            # delete data/maxsat/wms_crafted (even though it is not empty)
            import shutil

            shutil.rmtree("data/maxsat/wms_crafted")
        # delete .tgz file
        pathlib.Path("data/maxsat/wms_crafted.tgz").unlink()
        logging.info("Data extracted!")


def download_maxsat125_data():
    if not pathlib.Path(
        "data/maxsat/cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf"
    ).exists():
        logging.info(
            "cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf not found. Downloading..."
        )
        import gdown

        # download https://drive.google.com/file/d/1etCcts1icVJqD8KPsCH2rlHAOkf-ucj2
        url = "https://drive.google.com/uc?id=1etCcts1icVJqD8KPsCH2rlHAOkf-ucj2"
        logging.info(f"Downloading {url}")

        gdown.download(url, "data/maxsat/ce.zip", quiet=False)

        import zipfile

        with zipfile.ZipFile("data/maxsat/ce.zip", "r") as zip_ref:
            zip_ref.extractall("data/maxsat")

        # extract data/maxsat/mse18-new/cluster-expansion/benchmarks/IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz
        import gzip, shutil

        with gzip.open(
            "data/maxsat/mse18-new/cluster-expansion/benchmarks/IS1_5.0.5.0.0.5_softer_periodic.wcnf.gz",
            "rb",
        ) as f_in:
            # save to data/maxsat/cluster-expansion-IS1_5.wcnf
            with open(
                "data/maxsat/cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf",
                "wb",
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

        shutil.rmtree("data/maxsat/mse18-new")

        # delete .zip file
        pathlib.Path("data/maxsat/ce.zip").unlink()
        logging.info("Data extracted!")


def load_uci_data(
    n_features: Optional[int] = None,
):
    # taken from the BODi paper (https://arxiv.org/pdf/2303.01774.pdf)

    if not pathlib.Path("data/slice_localization_data.csv").exists():
        download_uci_data()

    try:
        path = pathlib.Path("data/slice_localization_data.csv").resolve()
        df = pd.read_csv(path, sep=",")
        data = df.to_numpy()

    except:
        raise FileNotFoundError(
            "The UCI slice_localization_data.csv dataset was not found. Please download it using the download_uci_data() function."
        )

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
        :10_000
    ]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    if n_features is not None:
        # Use Xgboost to figure out feature importances and keep only the most important features
        xgb = XGBRegressor(max_depth=8).fit(X, y)
        inds = (-xgb.feature_importances_).argsort()
        X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y
