import matplotlib
matplotlib.use('Agg')

import utils
import argparse
from openimages import OpenImages

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', dest='path',
                        help='path to openimages data', 
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()
data_dir = args.path
annotation_dir = '{}{}'.format(data_dir, 'annotations')

for subset in ['val', 'test', 'train']:
    print('converting {} data'.format(subset))
    # Select corresponding image directory
    image_dir = '{}{}'.format(data_dir, subset)
    # Convert annotations
    utils.convert_openimages_subset(annotation_dir, image_dir, subset)