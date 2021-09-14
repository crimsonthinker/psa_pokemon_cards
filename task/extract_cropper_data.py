import argparse
from task.loaders import UNETPreProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_img_height", type=int, default=3147, nargs='?',
        help="Origin image height for the training session")
    parser.add_argument("--origin_img_width", type=int, default=1860, nargs='?',
        help="Original image width for the training session")
    args = parser.parse_args()

    processor = UNETPreProcessor(
        args.origin_img_height,
        args.origin_img_width,
        3
    )
    processor.load()
    processor.operate()

