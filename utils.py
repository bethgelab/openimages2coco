import os
import imagesize

from tqdm import tqdm
from collections import defaultdict

def _url_to_license(licenses, mode='http'):
    # create dict with license urls as 
    # mode is either http or https
    
    # create dict
    licenses_by_url = {}

    for license in licenses:
        # Get URL
        if mode == 'https':
            url = 'https:' + license['url'][5:]
        else:
            url = license['url']
        # Add to dict
        licenses_by_url[url] = license
        
    return licenses_by_url

def convert_category_annotations(orginal_category_info):
    
    categories = []
    num_categories = len(orginal_category_info)
    for i in range(num_categories):
        cat = {}
        cat['id'] = i + 1
        cat['name'] = orginal_category_info[i][1]
        cat['freebase_id'] = orginal_category_info[i][0]
        
        categories.append(cat)
    
    return categories

def convert_image_annotations(original_image_metadata, original_image_annotations, image_dir, categories, licenses, verbose=1):
    
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    # Get dict with license urls
    licenses_by_url_http = _url_to_license(licenses, mode='http')
    licenses_by_url_https = _url_to_license(licenses, mode='https')
    
    # convert original image annotations to dict
    pos_img_lvl_anns = defaultdict(list)
    neg_img_lvl_anns = defaultdict(list)
    for ann in original_image_annotations[1:]:
        cat_of_ann = cats_by_freebase_id[ann[2]]['id']
        if int(ann[3]) == 1:
            pos_img_lvl_anns[ann[0]].append(cat_of_ann)
        elif int(ann[3]) == 0:
            neg_img_lvl_anns[ann[0]].append(cat_of_ann)
    
    #Create list
    images = []

    # loop through entries skipping title line
    num_images = len(original_image_metadata)
    for i in tqdm(range(1,num_images), mininterval=0.5):
        # Select image ID as key
        key = original_image_metadata[i][0]
        
        # Copy information
        img = {}
        img['id'] = key
        img['file_name'] = key + '.jpg'
        img['original_url'] = original_image_metadata[i][2]
        img['neg_category_ids'] = neg_img_lvl_anns.get(key, [])
        img['pos_category_ids'] = pos_img_lvl_anns.get(key, [])
        license_url = original_image_metadata[i][4]
        # Look up license id
        try:
            img['license'] = licenses_by_url_https[license_url]['id']
        except:
            img['license'] = licenses_by_url_http[license_url]['id']

        # Extract height and width
        if not os.path.exists(image_dir):
            filename = os.path.join(image_dir + "_"+img['file_name'][0].lower(), img['file_name'])
        else:
            filename = os.path.join(image_dir, img['file_name'])


        img['width'], img['height'] = imagesize.get(filename)
            
        # Add to list of images
        images.append(img)
        
    return images


def convert_instance_annotations(original_annotations, images, categories, start_index=0):
    
    imgs = {img['id']: img for img in images}
    cats = {cat['id']: cat for cat in categories}
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    annotations = []

    num_instances = len(original_annotations)
    for i in tqdm(range(1,num_instances), mininterval=0.5):
        # set individual instance id
        # use start_index to separate indices between dataset splits
        key = i + start_index
        csv_line = i
        ann = {}
        ann['id'] = key
        image_id = original_annotations[csv_line][0]
        ann['image_id'] = image_id
        ann['original_category_id'] = original_annotations[csv_line][2]
        ann['category_id'] = cats_by_freebase_id[original_annotations[csv_line][2]]['id']
        x = float(original_annotations[csv_line][4]) * imgs[image_id]['width']
        y = float(original_annotations[csv_line][6]) * imgs[image_id]['height']
        dx = (float(original_annotations[csv_line][5]) - float(original_annotations[csv_line][4])) * imgs[image_id]['width']
        dy = (float(original_annotations[csv_line][7]) - float(original_annotations[csv_line][6])) * imgs[image_id]['height']


        ann['bbox'] = [round(a, 2) for a in [x , y, dx, dy]]
        ann['area'] = round(dx * dy, 2)
        ann['isoccluded'] = int(original_annotations[csv_line][8])
        ann['istruncated'] = int(original_annotations[csv_line][9])
        ann['iscrowd'] = int(original_annotations[csv_line][10])
        ann['isdepiction'] = int(original_annotations[csv_line][11])
        ann['isinside'] = int(original_annotations[csv_line][12])
        annotations.append(ann)
        
    return annotations
