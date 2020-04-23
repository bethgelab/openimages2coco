import os
import csv
import json
import utils
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', '--path', dest='path',
                        help='path to openimages data', 
                        type=str)
    parser.add_argument('--version', default=6, type=int, help='Open Images Version')
    args = parser.parse_args()
    return args

args = parse_args()
base_dir = args.path

for subset in ['val', 'test', 'train']:
    # Convert annotations
    version = args.version
    print('converting {} data'.format(subset))
    
    # Select correct source files for each subset
    category_sourcefile = 'class-descriptions-boxable.csv'
    if subset == 'train':
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        annotation_sourcefile = 'train-annotations-bbox.csv'
        if args.version == 6:
            annotation_sourcefile = 'oidv6-train-annotations-bbox.csv'
        else:
            image_label_sourcefile = 'train-annotations-human-imagelabels-boxable.csv'
    elif subset == 'val':
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'validation-annotations-bbox.csv'
        image_label_sourcefile = 'validation-annotations-human-imagelabels-boxable.csv'
    elif subset == 'test':
        image_sourcefile = 'test-images-with-rotation.csv'
        annotation_sourcefile = 'test-annotations-bbox.csv'
        image_label_sourcefile = 'test-annotations-human-imagelabels-boxable.csv'

    # Load original annotations
    print('loading original annotations ...', end='\r')
    with open('{}/annotations/{}'.format(base_dir, category_sourcefile), 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        original_category_info = []
        for row in csv_f:
            original_category_info.append(row)

    with open('{}/annotations/{}'.format(base_dir, image_sourcefile), 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        original_image_metadata = []
        for row in csv_f:
            original_image_metadata.append(row)

    with open('{}/annotations/{}'.format(base_dir, annotation_sourcefile), 'r') as f:
        csv_f = csv.reader(f)
        original_annotations = []
        for row in csv_f:
            original_annotations.append(row)

    with open('{}/annotations/{}'.format(base_dir, image_label_sourcefile), 'r', encoding='utf-8') as f:
        csv_f = csv.reader(f)
        original_image_annotations = []
        for row in csv_f:
            original_image_annotations.append(row)
    print('loading original annotations ... Done')

    oi = {}

    # Add basic dataset info
    print('adding basic dataset info')
    oi['info'] = {'contributos': 'Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik',
                  'description': 'Open Images Dataset v{}'.format(version),
                  'url': 'https://storage.googleapis.com/openimages/web/index.html',
                  'version': '{:.1f}'.format(version),
                  'year': 2020}

    # Add license information
    print('adding basic license info')
    oi['licenses'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'},
                      {'id': 2, 'name': 'Attribution-NonCommercial License', 'url': 'http://creativecommons.org/licenses/by-nc/2.0/'},
                      {'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/'},
                      {'id': 4, 'name': 'Attribution License', 'url': 'http://creativecommons.org/licenses/by/2.0/'},
                      {'id': 5, 'name': 'Attribution-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-sa/2.0/'},
                      {'id': 6, 'name': 'Attribution-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nd/2.0/'},
                      {'id': 7, 'name': 'No known copyright restrictions', 'url': 'http://flickr.com/commons/usage/'},
                      {'id': 8, 'name': 'United States Government Work', 'url': 'http://www.usa.gov/copyright.shtml'}]

    # Convert category information
    print('converting category info')
    oi['categories'] = utils.convert_category_annotations(original_category_info)

    # Convert image mnetadata
    print('converting image info ...')
    image_dir = os.path.join(base_dir, subset)
    oi['images'] = utils.convert_image_annotations(original_image_metadata, original_image_annotations,
                                                   image_dir, oi['categories'], oi['licenses'])

    # Convert instance annotations
    print('converting annotations ...')
    oi['annotations'] = utils.convert_instance_annotations(original_annotations, oi['images'], oi['categories'], start_index=0)

    # Write annotations into .json file
    filename = os.path.join(base_dir, 'annotations', 'openimages_v{:d}_{}_bbox.json'.format(version, subset))
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))
    print('Done')