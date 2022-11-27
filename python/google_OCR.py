import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Google OCR')
    parser.add_argument('--img_dir', type=Path, default="data/scores")
    parser.add_argument('--out_dir', type=Path, default="data/scores_ocr")
    args = parser.parse_args()
    return args


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return texts


def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)
    all_img_paths = args.img_dir.glob("*.png")
    for img_path in all_img_paths:
        print("Processing", str(img_path))
        texts = detect_text(img_path)
        out_path = args.out_dir / (img_path.stem + ".txt")

        with open(out_path, "w", encoding="utf8") as f:
            for text in texts:
                t = text.description.encode('cp932', "ignore")
                t = t.decode('cp932')
                vertices = [
                    f"{vertex.x}, {vertex.y}" for vertex in text.bounding_poly.vertices]
                f.write(f"{t}, {','.join(vertices)}\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
