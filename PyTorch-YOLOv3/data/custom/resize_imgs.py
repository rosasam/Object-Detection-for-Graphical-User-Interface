from PIL import Image
import glob
from tqdm import tqdm

for img in tqdm(glob.glob('images/*jpg')):
    im = Image.open(img)
    im.thumbnail((416, 416))
    im.save(img)

