import os
import numpy as np
import json
import cv2
import pdb

# Use the same script for MOT16
# DATA_PATH = '../../data/mot16/'
DATA_PATH = 'dataset/'
OUT_PATH = 'dataset/'
SPLITS = ["TrainReal", "TestReal"]  # ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':
    for split in SPLITS:
        out_path = os.path.join(OUT_PATH, split, "annotations/", '{}.json'.format(split))
        data_path = os.path.join(DATA_PATH, split, "images/")
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'pedestrain'}],
               'videos': []}
        seqs = os.listdir(data_path)  # seqs are folders containing sequences of images
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            # Process Tracking Data
            if '.DS_Store' in seq:
                continue
            # if 'mot17' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
            #     continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            # img_path = seq_path + 'img1/'
            ann_path = os.path.join(data_path, "..", "annotations", f"{seq}.csv")  # seq_path + 'gt/gt.txt'
            images = os.listdir(seq_path)  # os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            # if HALF_VIDEO and ('half' in split):
            #     image_range = [0, num_images // 2] if 'train' in split else \
            #         [num_images // 2 + 1, num_images - 1]
            # else:
            #     image_range = [0, num_images - 1]
            image_range = [0, num_images - 1]
            for i in range(num_images):
                if (i < image_range[0] or i > image_range[1]):  # should never happen for us
                    continue
                image_info = {'file_name': '{}/{}_{:010d}.jpg'.format(seq, seq, i),
                              'id': image_cnt + i + 1,  # image id across all frames in the dataset
                              'frame_id': i + 1 - image_range[0],  # frame id within the sequence
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            # Process Detection Data
            if split != 'test':
                # det_path = seq_path + 'det/det.txt'
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                # dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                # if CREATE_SPLITTED_ANN and ('half' in split):
                #     anns_out = np.array([anns[i] for i in range(anns.shape[0]) if \
                #                          int(anns[i][0]) - 1 >= image_range[0] and \
                #                          int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                #     anns_out[:, 0] -= image_range[0]
                #     gt_out = seq_path + '/gt/gt_{}.txt'.format(split)
                #     fout = open(gt_out, 'w')
                #     for o in anns_out:
                #         fout.write(
                #             '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                #                 int(o[0]),int(o[1]),int(o[2]),int(o[3]),int(o[4]),int(o[5]),
                #                 int(o[6]),int(o[7]),o[8]))
                #     fout.close()
                # if CREATE_SPLITTED_DET and ('half' in split):
                #     dets_out = np.array([dets[i] for i in range(dets.shape[0]) if \
                #                          int(dets[i][0]) - 1 >= image_range[0] and \
                #                          int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                #     dets_out[:, 0] -= image_range[0]
                #     det_out = seq_path + '/det/det_{}.txt'.format(split)
                #     dout = open(det_out, 'w')
                #     for o in dets_out:
                #         dout.write(
                #             '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                #                 int(o[0]),int(o[1]),float(o[2]),float(o[3]),float(o[4]),float(o[5]),
                #                 float(o[6])))
                #     dout.close()

                print(' {} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if (frame_id - 1 < image_range[0] or frame_id - 1> image_range[1]):
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    if not ('15' in DATA_PATH):
                        if not (float(anns[i][8]) >= 0.25):
                            continue
                        if not (int(anns[i][6]) == 1):
                            continue
                        if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]): # Non-person
                            continue
                        if (int(anns[i][7]) in [2, 7, 8, 12]): # Ignored person
                            category_id = -1
                        else:
                            category_id = 1
                    else:
                        category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6])}
                    out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
