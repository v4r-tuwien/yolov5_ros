from tqdm import tqdm

from utils.general import Path, check_requirements, download, np, xyxy2xywhn

check_requirements(('pycocotools>=2.0',))
from pycocotools.coco import COCO



path="../datasets/Objects365"  # dataset root dir
train="images/train"  # train images (relative to 'path') 1742289 images
val="images/val" # val images (relative to 'path') 80000 images

# classes filter
classes = [44, 157, 205, 299, 343, 54, 256, 169, 283, 306, 357, 170, 197, 297, 316, 346, 123, 231, 286, 281, 242, 292, 327, 84, 269, 10, 8, 7]

# Make Directories
dir = Path(path)  # dataset root dir
for p in 'images', 'labels':
    (dir / p).mkdir(parents=True, exist_ok=True)
    for q in 'train', 'val':
        (dir / p / q).mkdir(parents=True, exist_ok=True)

# Train, Val Splits
for split, patches in [('train', 50 + 1), ('val', 43 + 1)]:
#for split, patches in [('val', 43 + 1)]:
    print(f"Processing {split} in {patches} patches ...")
    images, labels = dir / 'images' / split, dir / 'labels' / split

    # Move
    for f in tqdm(images.rglob('*.jpg'), desc=f'Moving {split} images'):
        f.rename(images / f.name)  # move to /images/{split}

    # Labels
    coco = COCO(dir / f'zhiyuan_objv2_{split}.json')
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    for cid, cat in enumerate(names):
        if cid in classes:
            print("cid: ", cid)
            print("cat: ", cat)
            print()
            catIds = coco.getCatIds(catNms=[cat])
            imgIds = coco.getImgIds(catIds=catIds)
            for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
                width, height = im["width"], im["height"]
                path = Path(im["file_name"])  # image filename
                try:
                    with open(labels / path.with_suffix('.txt').name, 'a') as file:
                        annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                        for a in coco.loadAnns(annIds):
                            x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                            xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                            x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                            file.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
                except Exception as e:
                    print(e)