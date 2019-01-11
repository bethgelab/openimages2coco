from pycocotools.coco import COCO

class OpenImages(COCO):
    
    def getOrigCats(self, cats=None):
        origCats = {}
        if not cats:
            cats = self.cats
        for i in cats.keys():
            key = cats[i]['original_id']
            origCats[key] = cats[i]

        self.origCats = origCats