import argparse
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="manual pose annotation")
    parser.add_argument("--html_dir", type=Path, default="data/nizisanzi-hai")
    args = parser.parse_args()
    return args


def process(html_path):
    all_image_paths = Path(
        r"C:\Users\furag\Documents\prog\MK8DXCountTool\MK8DXCountTool\data\images").glob("*.png")

    def convert_name(n):
        if n == "新レインボーロード":
            return "レインボーロード"
        return n

    count_dict = {convert_name(p.stem): 0 for p in all_image_paths}

    with open(html_path, encoding="utf8") as f:
        html = f.read()
        html = html.replace(' ', '')
        start = ";</span>1st<span"
        end = "&gt;</span>1位<spa"

        index = 0
        while True:
            s = html.find(start, index)
            if s < 0:
                break
            e = html.find(end, s)
            index = e

            debug = []
            for c in count_dict.keys():
                cnt = html.count(">" + c + "<", s, e)
                count_dict[c] += cnt
                if cnt >= 1:
                    debug.append((c, cnt))

            sum = np.sum([v for _, v in count_dict.items()])
#            print(sum, debug)

    return count_dict


def main(args):
    all_html_paths = list(args.html_dir.glob("*.html"))

    # check

    total_count_dict = {
        "ワリオこうざん": 3,
        "ウォーターパーク": 2,
        "ドラゴンロード": 2,
        "ハイラルサーキット": 2,
        "ベビィパーク": 2,
        "レインボーロード": 2,
        "どうぶつの森": 1,
        "キノピオハーバー": 1,
        "グラグラかざん": 1,
        "サンシャインくうこう": 1,
        "ツルツルツイスター": 1,
        "ドルフィンみさき": 1,
        "ピーチサーキット": 1,
        "GBAマリオサーキット": 1,
        "N64レインボーロード": 1,
    }

    for i, html_path in enumerate(all_html_paths):
        print(f"[{i+1}/{len(all_html_paths)}] {str(html_path)}")
        count_dict = process(html_path)
        for c, v in count_dict.items():
            total_count_dict.setdefault(c, 0)
            total_count_dict[c] += v
        print(np.sum([v for _, v in count_dict.items()]))

    ls = sorted([(c, v)
                for c, v in total_count_dict.items()], key=lambda p: -p[1])
    sum = np.sum([v for _, v in ls])
    print("sum=", sum)
    for c, v in ls:
        if v >= 1:
            print(f"{c}: {v} ({100 * v / sum:.1f}%)")


if __name__ == "__main__":
    args = parse_args()
    main(args)
