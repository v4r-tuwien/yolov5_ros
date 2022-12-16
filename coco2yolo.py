import json
import os

# Load the COCO annotation file
with open('coco_annotations.json', 'r') as f:
    coco = json.load(f)

# Extract the annotations and image filenames
annotations = coco['annotations']
images = coco['images']

# Create a mapping from image id to image filename
image_id_to_filename = {image['id']: image['file_name'] for image in images}

# Iterate through the COCO annotations
for annotation in annotations:
    # Extract the image id, bounding box coordinates, and class label
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    xmin = bbox[0]
    ymin = bbox[1]
    width = bbox[2]
    height = bbox[3]
    category_id = annotation['category_id']

    # Convert the bounding box coordinates to the YOLOv5 format
    xmax = xmin + width
    ymax = ymin + height
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    yolov5_bbox = [xcenter / width, ycenter / height, width, height]

    # Get the filename for the image
    filename = image_id_to_filename[image_id]

    # Save the YOLOv5 annotation to a file
    with open(f"{filename}.txt", 'a') as f:
        f.write(f"{category_id} {xcenter} {ycenter} {width} {height}\n")
