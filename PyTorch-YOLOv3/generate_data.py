import json
import glob
import sys, os
from copy import copy
import dicttoxml
from xml.etree.ElementTree import tostring
from xml.dom import minidom
from shutil import copyfile
import tensorflow as tf
import hashlib
from tqdm import tqdm
import random
from PIL import Image

IMG_SIZE = 416
#ANNOTATIONS_SAVE_DIR = 'annotations'
OUTPUT_PATH = './enrico.tfrecord'
BASE_PATH = '../../enrico'
BOUNDS_MAX_WIDTH = 1440
BOUNDS_MAX_HEIGHT = 2560

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

CLASSES = [
    'Multi-Tab',
    'Card',
    'Video',
    'Background Image',
    'Number Stepper',
    'Radio Button',
    'Checkbox',
    'Date Picker',
    'List Item',
    'On/Off Switch',
    'Slider',
    'Web View',
    'Text Button',
     #'Text',
    'Map View',
    'Drawer',
    'Modal',
    'Bottom Navigation',
    'Image',
    'Input',
    'Icon',
    'Advertisement',
    'Button Bar',
    'Toolbar',
    'Pager Indicator',
]

large_components = 0

# I/O
def load_hierarchies():
    directory = os.path.join(BASE_PATH, 'hierarchies')
    hierarchies = get_files(directory, 'json')
    # Filter out bad samples from the dataset
    hierarchies = [h for h in hierarchies if get_id(h) not in issues]
    data = []
    for pathname in hierarchies:
        with open(pathname) as f:
            hierarchy = json.load(f)
            hierarchy['id'] = get_id(pathname)
            data.append(hierarchy)
    return data

def load_screenshots():
    directory = os.path.join(BASE_PATH, 'screenshots')
    images = get_files(directory, 'jpg')
    images = [i for i in images if get_id(i) not in issues]
    data = []
    for pathname in images:
        with open(pathname) as f:
            image = json.load(f)
            data.append(hierarchy)
    return data

def get_files(directory, ending):
    return sorted(glob.glob(os.path.join(directory, f'*.{ending}')))

def get_id(filename):
    return os.path.basename(filename).split('.')[0]

def recursive_extract(hierarchy, hierarchy_id):
    children = hierarchy.get('children')
    components = []
    if children:
        for child in children:
            components.extend(recursive_extract(child, hierarchy_id))
    if is_component(hierarchy):
        components.append(create_component(hierarchy, hierarchy_id))
    return components

def create_component(dictionary, hierarchy_id):
    return {
        'class': dictionary['componentLabel'],
        'bounds': normalize_bounds(dictionary['bounds']),
        'image_id': hierarchy_id
    }

def is_component(dictionary):
    return 'componentLabel' in dictionary and dictionary['componentLabel'] in CLASSES

# NOTE: Normalizes to [0, 1] range, not training image size range 
def normalize_bounds(bounds):
    bounds = copy(bounds)
    for i in range(4):
        # Screenshots are transformed to squares, add the extra width and
        # height to bounds
        if i % 2 == 0:
            bounds[i] += (BOUNDS_MAX_HEIGHT - BOUNDS_MAX_WIDTH) // 2
            
        # uneven index: y, even index: x
        # divisor = BOUNDS_MAX_HEIGHT if i % 2 else BOUNDS_MAX_WIDTH
        # multiplier = IMG_HEIGHT if i % 2 else IMG_WIDTH
        # bounds[i] = bounds[i] * multiplier // divisor
        divisor = BOUNDS_MAX_HEIGHT # assuming height == width
        bounds[i] = bounds[i] / divisor
    return bounds

def filter_components_by_class(components, class_name):
    return [c for c in components if c['class'] == class_name]

def prettify(xml):
    rough_string = xml.decode('utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def component_to_VOC(component):
    '''
    Converts input component to XML adhering to Pascal VOC format.
    '''
    filename = f"{component['image_id']}.jpg"
    xmin, ymin, xmax, ymax = component['bounds']
    component = {
        'folder': ANNOTATIONS_SAVE_DIR,
        'filename': filename,
        'path': os.path.join(ANNOTATIONS_SAVE_DIR, filename),
        'source': {'database': 'Enrico'},
        'size': {
            'width': IMG_SIZE,
            'height': IMG_SIZE,
            'depth': 3
        },
        'segmented': 0,
        'object': {
            'name': component['class'],
            'truncated': 0,
            'difficult': 0,
            'bndbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
        }
    }
    return dicttoxml.dicttoxml(component, custom_root='annotation')

def component_group_to_tfrecord(component_group):
    class_map = {name: index for index, name in enumerate(CLASSES)}
    component = component_group[0]
    filename = f"{component['image_id']}.jpg"
    img_path = os.path.join(BASE_PATH, 'screenshots', filename)
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    height = IMG_GEIGHT
    width = IMG_WIDTH

    xmin, ymin, xmax, ymax = [[],[],[],[]]
    classes = []
    classes_text = []
    difficult_obj = []
    truncated = []
    views = []
    
    for component in component_group:
        if component['class'] not in CLASSES:
            continue
        obj_xmin, obj_ymin, obj_xmax, obj_ymax = component['bounds']
        xmin.append(float(obj_xmin) / width)
        ymin.append(float(obj_ymin) / height)
        xmax.append(float(obj_xmax) / width)
        ymax.append(float(obj_ymax) / height)
        classes_text.append(component['class'].encode('utf8'))
        classes.append(class_map[component['class']]) # FIX
        difficult_obj.append(0)
        truncated.append(0)
        #views.append(obj['pose'].encode('utf8'))
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        #'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example

def component_group_to_txt(component_group):
    name = f"{component_group[0]['image_id']}.txt"
    large_components = 0
    output_list = []
    for component in component_group:
        label_index = CLASSES.index(component['class'])
        x_min, y_min, x_max, y_max = component['bounds']
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        #if width > 0.9 and height > 0.9:
        if x_min < 0.01 or y_min < 0.005 or x_max > 0.95 or y_max > 0.99 or width > 0.99 or height > 0.99:
            large_components += 1 
            continue
        output_list.append(' '.join([str(item) for item in [label_index, x_center, y_center, width, height]]))
    return (name, '\n'.join(output_list), large_components)
    
def split_data(data, training_size=0.75):
    data = [d for d in data]
    random.shuffle(data)
    training_len = (len(data)*int(training_size*100)//100)
    return data[:training_len], data[training_len:]

def save_txt(txt, path):
    with open(path, 'w') as outfile:
        outfile.write(txt)

def save_xml(xml, path):
    with open(path, 'wb') as outfile:
        outfile.write(xml)

def print_component_stats(components):
    print(f'TOTAL: {len(components)}')
    classes = [c['class'] for c in components]
    for c in list(set(classes)):
        print(f'{c}: {classes.count(c)}')

def copy_component_image(component):
    filename = f'{component["image_id"]}.jpg'
    src = os.path.join(BASE_PATH, 'screenshots', filename)
    dst = os.path.join('./data/custom/images', filename)

    background = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 'white')
    picture = Image.open(src).convert('RGB')
    picture.thumbnail((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    background.paste(picture, ((IMG_SIZE - picture.size[0]) // 2, 0))

    background.save(dst, 'JPEG')
    

annotation_path = './data/custom/labels/'
training_set_path = './data/custom/'

hi = load_hierarchies()
#img = load_screenshots()
components = [recursive_extract(h, h['id']) for h in hi]

files = glob.glob('./data/custom/images/*jpg')
for f in files:
    os.remove(f)
for c in tqdm(components):
    copy_component_image(c[0])

txts = [component_group_to_txt(component_group) for component_group in components if component_group]
large_components = sum([t[2] for t in txts])
txts = [(t[0], t[1]) for t in txts]
txt_paths = ['./data/custom/images/' + txt[0].split('.')[0] + '.jpg' for txt in txts]
train_data, validation_data = split_data(txt_paths)

save_txt('\n'.join(train_data), training_set_path + 'train.txt')
save_txt('\n'.join(validation_data), training_set_path + 'valid.txt')
for txt in tqdm(txts):
    save_txt(txt[1], annotation_path + txt[0]) 
print(f'large_components: {large_components}')
print(f'Nof classes: {len(CLASSES)}')
print('DONE')
