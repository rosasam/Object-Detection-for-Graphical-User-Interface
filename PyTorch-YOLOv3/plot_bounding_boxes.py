import argparse
import os
from PIL import Image, ImageDraw
from math import floor
import glob

IMG_DIR = 'data/custom/images'
LABELS_DIR = 'data/custom/labels'
IMG_SIZE = 416

issues = [
    '22712',
    '55591',
    '39813',
    '2251',
    '1411',
    '2030',
    '57954',
    '23875',
    '65428',
    '39254',
    '50105',
    '50109',
    '72201',
    '19141',
    '1986',
    '68484',
    '2081',
    '38375',
    '41602',
    '71046',
    '37707',
    '21227',
    '6105',
    '37021',
    '31002',
    '10498',
    '56375',
    '63070',
    '32802',
    '526',
    '49581',
    '23788',
    '51032',
    '14285',
    '63183',
    '28511',
    '71558',
    '19713',
    '71999',
    '16140',
    '41458',
    '59514',
    '1130',
    '1753',
    '2559',
    '3140',
    '3993',
    '4068',
    '4457',
    '4833',
    '8522',
    '9091',
    '9371',
    '10798',
    '13721',
    '17018',
    '17169',
    '17254',
    '17921',
    '18262',
    '20476',
    '22753',
    '22759',
    '23622',
    '24269',
    '27717',
    '27884',
    '29573',
    '29865',
    '29983',
    '29992',
    '30007',
    '30026',
    '30929',
    '54918',
    '60754',
    '6227',
    '64395',
    '66899',
    '67983',
    '68492'
]

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

def render_bbs(image, bbs, id):
    draw = ImageDraw.Draw(image)
    for bb in bbs:
        draw.rectangle((bb), outline='blue', width=2)
    image.save(f'check_enrico_images/{id}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ids', type=str, nargs='*')

    opt = parser.parse_args()
    ids = opt.ids if opt.ids else [path.split('/')[-1].split('.')[0] for path in glob.glob(f'{IMG_DIR}/*.jpg')]

    for rico_id in ids:
        if rico_id in issues:
            continue
        imgpath = os.path.join(IMG_DIR, f'{rico_id}.jpg')
        labelpath = os.path.join(LABELS_DIR, f'{rico_id}.txt')

        bounding_boxes = read_labels(labelpath)
        image = read_image(imgpath)

        render_bbs(image, bounding_boxes, rico_id)
