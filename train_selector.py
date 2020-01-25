from mas import train_xl_selector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        required=True
    )
    parser.add_argument(
        "--eval-path",
        required=True
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.001,
        required=True
    )
    args = parser.parse_args()

    train_xl_selector(args)