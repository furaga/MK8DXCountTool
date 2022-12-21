import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Google OCR')
    # parser.add_argument('--img_dir', type=Path, default="data/scores")
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--out_dir', type=Path, default="out")
    parser.add_argument('--dt', type=float, default=0.1)
    args = parser.parse_args()
    return args


def main(args):
    import time
    import mss

    since = time.time()
    i_frame = 0
    with mss.mss() as mss_instance:
        while True:
            t = time.time() - since
            i_frame += 1
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / f"{i_frame:04d}F_{int(t * 1000):06d}ms.jpg"
            mss_instance.shot(mon=args.monitor, output=str(out_path))
            print("Saved", out_path)

            time_to_sleep = i_frame * args.dt - (time.time() - since)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == '__main__':
    args = parse_args()
    main(args)
