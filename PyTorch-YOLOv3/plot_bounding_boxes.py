import argparse
import os
from PIL import Image, ImageDraw
from math import floor

IMG_DIR = 'data/custom/images'
LABELS_DIR = 'data/custom/labels'
IMG_SIZE = 416

def read_labels(labelpath):
    with open(labelpath, 'r') as labelfile:
        lines = labelfile.readlines()
    return [parse_label(line) for line in lines]

def parse_label(label_line):
    _, x_center, y_center, width, height = [floor(float(label) * IMG_SIZE) for label in label_line.split(' ')]
    return [x_center - (width // 2), y_center - (height // 2), x_center + (width // 2), y_center + (height // 2)] 

def read_image(imgpath):
    # Paste image on square background
    background = Image.new('RGBA', (IMG_SIZE, IMG_SIZE))
    picture = Image.open(imgpath).convert('RGBA')
    background.paste(picture)
    return background

def render_bbs(image, bbs):
    draw = ImageDraw.Draw(image)
    for bb in bbs:
        draw.rectangle((bb), outline='blue', width=2)
    image.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ids', type=str, nargs='+')

    opt = parser.parse_args()

    for rico_id in opt.ids:
        imgpath = os.path.join(IMG_DIR, f'{rico_id}.jpg')
        labelpath = os.path.join(LABELS_DIR, f'{rico_id}.txt')

        bounding_boxes = read_labels(labelpath)
        image = read_image(imgpath)

        render_bbs(image, bounding_boxes)
