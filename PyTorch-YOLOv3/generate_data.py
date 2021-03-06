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

IMG_WIDTH = 540
IMG_HEIGHT = 960
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
    'Text',
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
    return 'componentLabel' in dictionary

# NOTE: Normalizes to [0, 1] range, not training image size range 
def normalize_bounds(bounds):
    bounds = copy(bounds)
    for i in range(4):
        # uneven index: y, even index: x
        # divisor = BOUNDS_MAX_HEIGHT if i % 2 else BOUNDS_MAX_WIDTH
        # multiplier = IMG_HEIGHT if i % 2 else IMG_WIDTH
        # bounds[i] = bounds[i] * multiplier // divisor
        divisor = IMG_HEIGHT if i % 2 else IMG_WIDTH
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
            'width': IMG_WIDTH,
            'height': IMG_HEIGHT,
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

    output_list = []
    for component in component_group:
        label_index = CLASSES.index(component['class'])
        x_min, y_min, x_max, y_max = component['bounds']
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        output_list.append(' '.join([str(item) for item in [label_index, x_center, y_center, width, height]]))
    
    return (name, '\n'.join(output_list))
    
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
    if not os.path.exists(dst):
        copyfile(src, dst)

annotation_path = './data/custom/labels/'
training_set_path = './data/custom/'

hi = load_hierarchies()
#img = load_screenshots()
components = [recursive_extract(h, h['id']) for h in hi]
# for c in tqdm(components):
#     copy_component_image(c[0]) 
txts = [component_group_to_txt(component_group) for component_group in components]
txt_paths = ['./data/custom/images/' + txt[0].split('.')[0] + '.jpg' for txt in txts]
train_data, validation_data = split_data(txt_paths)

save_txt('\n'.join(train_data), training_set_path + 'train.txt')
save_txt('\n'.join(validation_data), training_set_path + 'valid.txt')
for txt in tqdm(txts):
    save_txt(txt[1], annotation_path + txt[0]) 

# writer = tf.io.TFRecordWriter(OUTPUT_PATH)
# for component_group in tqdm.tqdm(components):
#     record = component_group_to_tfrecord(component_group) 
#     writer.write(record.SerializeToString())
# writer.close()
# components = [d for c in components for d in c]
# print_component_stats(components)

# # Only generate annotations for text buttons for now
# components = filter_components_by_class(components, 'Text Button')
# [copy_component_image(component) for component in components]
# xmls = [component_to_VOC(component) for component in components]

# for i, xml in enumerate(xmls):
#     save_xml(xml, os.path.join(ANNOTATIONS_SAVE_DIR, f'{i:05d}.xml'))

print('DONE')
