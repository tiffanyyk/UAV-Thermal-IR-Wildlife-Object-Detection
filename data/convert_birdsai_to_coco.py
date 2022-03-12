import os
import numpy as np
import json
from PIL import Image
import cv2
import pdb

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '/data/workspace/ROB498/data/dataset/'
OUT_PATH = '/data/workspace/ROB498/data/dataset/coco_format'
SPLITS = ["TrainReal", "TestReal"]  # ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == '__main__':
    for split in SPLITS:
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))  # split, "annotations/", '{}.json'.format(split))
        data_path = os.path.join(DATA_PATH, split, "images/")
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 9, 'name': 'unknown'},
                              {'id': 0, 'name': 'human'},
                              {'id': 1, 'name': 'elephant'},
                              {'id': 2, 'name': 'lion'},
                              {'id': 3, 'name': 'giraffe'},  # only in real data
                              {'id': 4, 'name': 'dog'},  # only in real data
                              {'id': 5, 'name': 'crocodile'},  # only in synthetic
                              {'id': 6, 'name': 'hippo'},  # only in synthetic
                              {'id': 7, 'name': 'zebra'},  # only in synthetic
                              {'id': 8, 'name': 'rhino'}],  # only in synthetic
               'videos': []}
        seqs = os.listdir(data_path)  # seqs are folders containing sequences of images
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            # Process Tracking Data
            if '.DS_Store' in seq:
                continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            ann_path = os.path.join(data_path, "..", "annotations", f"{seq}.csv")  # seq_path + 'gt/gt.txt'
            images = os.listdir(seq_path)  # os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            image_range = [0, num_images - 1]  # note that image numbering doesn't always start at 0
            for i, file_int in enumerate(sorted([int(image[-14:-4]) for image in images if 'jpg' in image])):
                # get image size
                img_file_name = '{}/{}_{:010d}.jpg'.format(seq, seq, file_int)
                image = Image.open(os.path.join(data_path, img_file_name))
                image = np.array(image)
                image_info = {'file_name': os.path.join(split, "images/", img_file_name),
                              'id': image_cnt + i + 1,  # image id across all frames in the dataset
                              'frame_id': i + 1 - image_range[0],  # frame id within the sequence
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'width': image.shape[1],
                              'height': image.shape[0]}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            # Process Detection Ground Truth
            if split != 'test':
                # for birdsai, anns is: <frame_number>, <object_id>, <x>, <y>, <w>, <h>, <class>, <species>, <occlusion>, <noise>
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

                print(' {} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    category_id = int(anns[i][7]) if int(anns[i][7]) >= 0 else 9  # birdsai species
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': int(1)}  # birdsai doesn't give confidence because boxes are ground truth annotations
                    out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
