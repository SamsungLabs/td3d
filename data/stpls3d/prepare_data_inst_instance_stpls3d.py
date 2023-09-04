# https://github.com/JonasSchult/Mask3D/blob/main/datasets/preprocessing/stpls3d_preprocessing.py
import numpy as np
import pandas as pd

import glob
import os
import pickle

def splitPointCloud(cloud, size=50.0, stride=50):
    limitMax = np.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limitMax[0] - size) / stride)) + 1
    depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)
    return blocks

def getFiles(files, fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        num = name[:2] if name[:2].isdigit() else name[:1]
        if int(num) in fileSplit:
            res.append(filePath)
    return res

def preparePthFiles(files, split, crop_size=50):

    os.makedirs("points", exist_ok=True)
    os.makedirs("instance_mask", exist_ok=True)
    os.makedirs("semantic_mask", exist_ok=True)
    

    data_info_list = []
    counter = 0

    for file in files:

        points = pd.read_csv(file, header=None).values
        name = os.path.basename(file).strip(".txt")
        points[:, :3] = points[:, :3] - points[:, :3].min(0)
        blocks = splitPointCloud(points, size=crop_size, stride=crop_size)

        for blockNum, block in enumerate(blocks):
            outFilePath = name + str(blockNum)

            coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))
            colors = np.ascontiguousarray(block[:, 3:6])
            coords = np.float32(coords)
            colors = np.float32(colors)
            points = np.hstack((coords, colors))

            sem_labels = np.ascontiguousarray(block[:, -2]).astype(np.int32)
            instance_labels = np.ascontiguousarray(block[:, -1]).astype(np.int32)

            assert sem_labels.min() >= 0
            assert np.all((instance_labels == -100) == (sem_labels == 0))
            
            instance_labels[instance_labels == -100] = -1            
            idxs = np.unique(instance_labels)

            if idxs[0] == -1:
                uniqueInstances = idxs[1:]
            else:
                uniqueInstances = idxs
            
            remapper_instance = np.ones(50000) * (-1)
            for i, j in enumerate(uniqueInstances):
                remapper_instance[j] = i

            instance_labels = remapper_instance[instance_labels]

            uniqueSemantics = (np.unique(sem_labels))[1:]

            if split == "train" and (
                len(uniqueInstances) < 10 or (len(uniqueSemantics) >= (len(uniqueInstances) - 2))
            ):
                print("unique insance: %d" % len(uniqueInstances))
                print("unique semantic: %d" % len(uniqueSemantics))
                print()
                counter += 1
            else:

                points_path = os.path.join('points', f'{outFilePath}.bin')
                semantic_path = os.path.join('semantic_mask', f'{outFilePath}.bin')
                instance_path = os.path.join('instance_mask', f'{outFilePath}.bin')
                
                points.astype(np.float32).tofile(points_path)
                sem_labels.astype(np.int64).tofile(semantic_path)
                instance_labels.astype(np.int64).tofile(instance_path)

                scene_info = {}
                scene_info['annos'] = {'gt_num': 0}
                scene_info['point_cloud'] = {'num_features': 6, 'lidar_idx': outFilePath}
                scene_info['pts_path'] = points_path
                scene_info['pts_instance_mask_path'] = instance_path
                scene_info['pts_semantic_mask_path'] = semantic_path 
                data_info_list.append(scene_info)

    print("Total skipped file :%d" % counter)
    return data_info_list


if __name__ == "__main__":
    data_folder = "Synthetic_v3_InstanceSegmentation"

    filesOri = sorted(glob.glob(data_folder + "/*.txt"))

    trainSplit = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    trainFiles = getFiles(filesOri, trainSplit)
    split = "train"
    data_info = preparePthFiles(trainFiles, split)
    with open('stpls3d_infos_train.pkl', 'wb') as file:
        pickle.dump(data_info, file)


    valSplit = [5, 10, 15, 20, 25]
    split = "val"
    valFiles = getFiles(filesOri, valSplit)
   
    data_info = preparePthFiles(valFiles, split)
    with open('stpls3d_infos_val.pkl', 'wb') as file:
        pickle.dump(data_info, file)
