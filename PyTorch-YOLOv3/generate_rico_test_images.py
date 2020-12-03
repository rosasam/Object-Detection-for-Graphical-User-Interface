from PIL import Image
import sys
import glob

def copy_image(filename, img_size):
    src = os.path.join('../../enrico', 'screenshots', filename)
    dst = os.path.join('./screenshot_test', filename)

    background = Image.new('RGB', (img_size, img_size))
    picture = Image.open(src).convert('RGB')
    picture.thumbnail((img_size, img_size), Image.ANTIALIAS)
    background.paste(picture, ((img_size - picture.size[0]) // 2, 0))

    background.save(dst, 'JPEG')

if __name__ == '__main__':
    if len(sys.argv) < 3:
      exit(0)
    size = sys.argv[1]
    n_images = sys.argv[2]
    images = glob.glob('../../enrico/screenshots/*.jpg')
    for image in images[:n]:
      copy_image(image, img_size)