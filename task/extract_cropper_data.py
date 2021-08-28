import argparse
from task.loaders import UNETPreProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_img_height", type=int, default=512, nargs='?',
        help="Image height for the training session")
    parser.add_argument("--origin_img_width", type=int, default=512, nargs='?',
        help="Image width for the training session")
    parser.add_argument("--dim", type=int, default=3, nargs='?',
        help="Image didmension for the training session")
    args = parser.parse_args()

    processor = UNETPreProcessor(
        args.origin_img_height,
        args.origin_img_width,
        args.dim
    )
    processor.load()
    processor.operate()

