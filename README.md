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
ptyhon convert.py PATH_TO_OPENIMAGES
```

### Output

The generated annotations can be loaded and used with the standard MS Coco tools:
```
from pycocotools.coco import COCO

# Example for the validation set
openimages = COCO('PATH_TO_OPENIMAGES/annotations/val-annotations-bbox.json')
```
