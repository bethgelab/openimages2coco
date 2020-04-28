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

def _list_to_dict(list_data):
    
    dict_data = []
    columns = list_data.pop(0)
    for i in range(len(list_data)):
        dict_data.append({columns[j]: list_data[i][j] for j in range(len(columns))})
                         
    return dict_data

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

def convert_image_annotations(original_image_metadata,
                              original_image_annotations,
                              original_image_sizes,
                              image_dir,
                              categories,
                              licenses,
                              origin_info=False):
    
    original_image_metadata_dict = _list_to_dict(original_image_metadata)
    original_image_annotations_dict = _list_to_dict(original_image_annotations)
    
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    if original_image_sizes:
        image_size_dict = {x[0]:  [int(x[1]), int(x[2])] for x in original_image_sizes[1:]}
    else:
        image_size_dict = {}
    
    # Get dict with license urls
    licenses_by_url_http = _url_to_license(licenses, mode='http')
    licenses_by_url_https = _url_to_license(licenses, mode='https')
    
    # convert original image annotations to dicts
    pos_img_lvl_anns = defaultdict(list)
    neg_img_lvl_anns = defaultdict(list)
    for ann in original_image_annotations_dict[1:]:
        cat_of_ann = cats_by_freebase_id[ann['LabelName']]['id']
        if int(ann['Confidence']) == 1:
            pos_img_lvl_anns[ann['ImageID']].append(cat_of_ann)
        elif int(ann['Confidence']) == 0:
            neg_img_lvl_anns[ann['ImageID']].append(cat_of_ann)
    
    #Create list
    images = []

    # loop through entries skipping title line
    num_images = len(original_image_metadata_dict)
    for i in tqdm(range(num_images), mininterval=0.5):
        # Select image ID as key
        key = original_image_metadata_dict[i]['ImageID']
        
        # Copy information
        img = {}
        img['id'] = key
        img['file_name'] = key + '.jpg'
        img['neg_category_ids'] = neg_img_lvl_anns.get(key, [])
        img['pos_category_ids'] = pos_img_lvl_anns.get(key, [])
        if origin_info:
            img['original_url'] = original_image_metadata_dict[i]['OriginalURL']
            license_url = original_image_metadata_dict[i]['License']
            # Look up license id
            try:
                img['license'] = licenses_by_url_https[license_url]['id']
            except:
                img['license'] = licenses_by_url_http[license_url]['id']

        # Extract height and width
        image_size = image_size_dict.get(key, None)
        if image_size is not None:
            img['width'], img['height'] = image_size
        else:
            filename = os.path.join(image_dir, img['file_name'])
            img['width'], img['height'] = imagesize.get(filename)
            
        # Add to list of images
        images.append(img)
        
    return images


def convert_instance_annotations(original_annotations, images, categories, start_index=0):
    
    original_annotations_dict = _list_to_dict(original_annotations)
    
    imgs = {img['id']: img for img in images}
    cats = {cat['id']: cat for cat in categories}
    cats_by_freebase_id = {cat['freebase_id']: cat for cat in categories}
    
    annotations = []
    
    annotated_attributes = [attr for attr in ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'] if attr in original_annotations[0]]

    num_instances = len(original_annotations_dict)
    for i in tqdm(range(num_instances), mininterval=0.5):
        # set individual instance id
        # use start_index to separate indices between dataset splits
        key = i + start_index
        csv_line = i
        ann = {}
        ann['id'] = key
        image_id = original_annotations_dict[csv_line]['ImageID']
        ann['image_id'] = image_id
        ann['freebase_id'] = original_annotations_dict[csv_line]['LabelName']
        ann['category_id'] = cats_by_freebase_id[ann['freebase_id']]['id']
        
        xmin = float(original_annotations_dict[csv_line]['XMin']) * imgs[image_id]['width']
        ymin = float(original_annotations_dict[csv_line]['YMin']) * imgs[image_id]['height']
        xmax = float(original_annotations_dict[csv_line]['XMax']) * imgs[image_id]['width']
        ymax = float(original_annotations_dict[csv_line]['YMax']) * imgs[image_id]['height']
        dx = xmax - xmin
        dy = ymax - ymin
        ann['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
        ann['area'] = round(dx * dy, 2)
        
        for attribute in annotated_attributes:
            ann[attribute.lower()] = int(original_annotations_dict[csv_line][attribute])

        annotations.append(ann)
        
    return annotations
