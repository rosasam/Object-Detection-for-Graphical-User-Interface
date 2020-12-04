from PIL import Image
import sys
import glob
import os

IMPORT_DIR = '../../my_screenshots' #'../../enrico/screenshots'

def copy_image(filename, img_size):
    src = os.path.join(IMPORT_DIR, filename)
    dst = os.path.join('./screenshot_test', filename)

    background = Image.new('RGB', (img_size, img_size))
    picture = Image.open(src).convert('RGB')
    picture.thumbnail((img_size, img_size), Image.ANTIALIAS)
    background.paste(picture, ((img_size - picture.size[0]) // 2, 0))

    background.save(dst, 'JPEG')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit(0)
    img_size = int(sys.argv[1])
    n_images = int(sys.argv[2])
    images = glob.glob(f'{IMPORT_DIR}/*.jpg')
    for image in images[:n_images]:
        copy_image(image.split('/')[-1], img_size)
