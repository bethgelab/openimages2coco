import matplotlib
matplotlib.use('Agg')

import utils
from openimages import OpenImages

data_dir = '/gpfs01/bethge/data/openimages/'
annotation_dir = '{}{}'.format(data_dir, 'annotations')

for subset in ['val', 'test', 'train']:
    print('converting {} data'.format(subset))
    # Select corresponding image directory
    image_dir = '{}{}'.format(data_dir, subset)
    # Convert annotations
    utils.convert_openimages_subset(annotation_dir, image_dir, subset)