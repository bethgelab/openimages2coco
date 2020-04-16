# openimages2coco
Convert [Open Images](https://storage.googleapis.com/openimages/web/index.html "Open Images Homepage") annotations into [MS Coco](http://cocodataset.org "MS Coco Homepage") format to make it a drop in replacement.

### Functionality

This conversion routine will load the original .csv annotation files form Open Images, convert the annotations into the list/dict based format of [MS Coco annotations](http://cocodataset.org/#format-data) and store them as a .json file in the same folder.

### Usage

Download the CocoAPI from https://github.com/cocodataset/cocoapi \
Install Coco API:
```
cd PATH_TO_COCOAPI/PythonAPI
make install
```

Download Open Images from https://storage.googleapis.com/openimages/web/download.html \
-> Store the images in three folders called: ```train, val and test``` \
-> Store the annotations for all three splits in a separate folder called: ```annotations```

Run conversion:
```
python3 convert.py -p PATH_TO_OPENIMAGES
```

### Output

The generated annotations can be loaded and used with the standard MS Coco tools:
```
from pycocotools.coco import COCO

# Example for the validation set
openimages = COCO('PATH_TO_OPENIMAGES/annotations/val-annotations-bbox.json')
```

### Issues
- The evaluation tools from the Coco API are not yet working with the converted Open Images annotations
- A few images are in a weird format that returns a list [image, some_annotations] when loaded with skimage.io.imread(image). These seem to be corrupted .jpg files and we will contact the data set developers on this issue. For the moment something like the following function can be used to catch all possible formatting issues including RGBA and monochrome images:
```
def load_image(self, image_id):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(self.image_info[image_id]['path'])
    # If image has additional annotations
    if image.ndim == 1:
        image = image[0]
    # If grayscale. Convert to RGB for consistency.
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image
```
