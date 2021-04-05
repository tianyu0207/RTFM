import numpy as np
import os
import glob
import numpy as np
from scipy.io import loadmat
from os import walk

'''
FOR UCF CRIME
'''
root_path = "/home/yu/yu_ssd/i3d_features_test/"
dirs = os.listdir(root_path)
rgb_list_file ='ucf-i3d-test.list'
temporal_root = '/home/yu/PycharmProjects/DeepMIL-master/list/Matlab_formate/'
mat_name_list = os.listdir(temporal_root)

file_list = list(open(rgb_list_file))
num_frame = 0
gt = []
for file in file_list:

    features = np.load(file.strip('\n'), allow_pickle=True)
    features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    num_frame = features.shape[0] * 16

    split_file = file.split('/')[-1].split('_')[0]
    mat_prefix = '_x264.mat'
    mat_file = split_file + mat_prefix
    # \features.shape)
    # print(num_frame)
    count = 0
    if 'Normal_' in file: # if it's normal
        # print('hello')
        for i in range(0, num_frame):
            gt.append(0.0)
            count+=1

    else: #if it's abnormal # get the name from temporal file


        if mat_file in mat_name_list:
            second_event = False
            annots = loadmat(os.path.join(temporal_root, mat_file))
            annots_idx = annots['Annotation_file']['Anno'].tolist()

            start_idx = annots_idx[0][0][0][0]
            end_idx = annots_idx[0][0][0][1]

            if len(annots_idx[0][0]) == 2:
                second_event = True
                # print(annots_idx)
            # if file is '/home/yu/yu_ssd/i3d_features_test/Assault006_x264_i3d.npy'
            # check if there's second events
            if not second_event:
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count +=1
                if not (end_idx + 1) > num_frame:
                    for i in range(start_idx, end_idx + 1):
                        gt.append(1.0)
                        count += 1
                    if (num_frame - end_idx) < 16:
                        for i in range(end_idx + 1, num_frame):
                            print('hello, last frame has abnormal event')
                            print(end_idx)
                            print(num_frame)
                            gt.append(1.0)
                            count += 1
                    else:
                        for i in range(end_idx+1, num_frame):
                            gt.append(0.0)
                            count += 1
                else:
                    for i in range(start_idx, end_idx):
                        gt.append(1.0)
                        print('end idx larger than number of frames')
                        print(end_idx)
                        print(num_frame)
                        count += 1



            else:
                start_idx_2 = annots_idx[0][0][1][0]
                end_idx_2 = annots_idx[0][0][1][1]
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count += 1
                for i in range(start_idx, end_idx + 1):
                    gt.append(1.0)
                    count += 1
                for i in range(end_idx+1, start_idx_2):
                    gt.append(0.0)
                    count += 1
                if not (end_idx_2 + 1) > num_frame:
                    for i in range(start_idx_2, end_idx_2 + 1):
                        gt.append(1.0)
                        count += 1
                    if (num_frame - end_idx_2) < 16:
                        for i in range(end_idx_2 + 1, num_frame):
                            print('hello, last clip has abnormal event')
                            print(end_idx_2)
                            print(num_frame)
                            gt.append(1.0)
                            count += 1
                    else:
                        for i in range(end_idx_2 + 1, num_frame):
                            gt.append(0.0)
                            count += 1
                else:
                    for i in range(start_idx_2, end_idx_2):
                        gt.append(1.0)
                        count += 1
                if count != num_frame:

                    print(annots_idx)
                    print(num_frame)
                    print(count)
                    print(end_idx_2 +1)


    if count != num_frame:
        print(file)
        exit(1)


            # print(annots_idx)
            # annots = annots.flatten()
            # ann = list(annots)
            # ann = np.squeeze(annots)
            # print(list(annots))
            # print(ann)
            # print(annots[0][0])



        # print('abnormal')
output_file = '/home/yu/PycharmProjects/DeepMIL-master/list/gt-ucf.npy'
gt = np.array(gt, dtype=float)
np.save(output_file, gt)
print(len(gt))





