import argparse
import os

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


def go(args):
    run = wandb.init(job_type="data_split")
    run.config.update(vars(args))

    # Download artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Stratify
    stratify = None
    if args.stratify_by.lower() != "none":
        stratify = df[args.stratify_by]

    # Split test
    trainval_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify,
    )

    # Split validation from remaining
    stratify_trainval = None
    if stratify is not None:
        stratify_trainval = trainval_df[args.stratify_by]

    # val_size is fraction of remaining set
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=args.val_size,
        random_state=args.random_seed,
        stratify=stratify_trainval,
    )

    # Save locally
    trainval_path = "trainval_data.csv"
    test_path = "test_data.csv"
    trainval_df.to_csv(trainval_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Log artifacts
    trainval_art = wandb.Artifact(args.trainval_artifact, type="trainval_data")
    trainval_art.add_file(trainval_path)
    run.log_artifact(trainval_art)

    test_art = wandb.Artifact(args.test_artifact, type="test_data")
    test_art.add_file(test_path)
    run.log_artifact(test_art)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split cleaned data into train/val and test sets")

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--test_artifact", type=str, required=True)

    args = parser.parse_args()
    go(args)