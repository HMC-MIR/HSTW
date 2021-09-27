import numpy as np
import pickle
import librosa
from scipy.spatial.distance import cdist
import python_speech_features
from sklearn.metrics.pairwise import euclidean_distances

sigma = np.array([[0, 1], [1, 0], [1, 1]])

bitrateKBPS = 160

dtw_weights = \
{'D1': np.array([1, 1, 1]),
'D2': np.array([0, 1, 1]),
'D3': np.array([1, 2, 1]),
'D4': np.array([1, 1, 2])}

#FROM 02
#def getMFCC(query_id, time = None, piece_type='reference', edit_type='n1', mfcc_type='old', save=False):
def getMFCC(query_id, time = None, piece_type='reference', edit_type='n1', mfcc_type='new', save=False):
    if piece_type =='reference':
        file_dir = '/mnt/data0/agoutam/TamperingDetection/speech/ref/wav/{}.wav'.format(query_id.replace("_160", ""))
    else:
        file_dir = '/mnt/data0/agoutam/TamperingDetection/speech/queries/wav/160kbps/{}sec/{}_{}_{}.wav'.format(time, query_id, edit_type, bitrateKBPS)
    if mfcc_type == 'old':
        y, sr = librosa.load(file_dir, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        return mfcc.T
    y, sr = librosa.load(file_dir, sr=16000)
    mfcc = python_speech_features.mfcc(y, sr, winstep = 0.01)
    delta_mfcc = python_speech_features.delta(mfcc, 2)
    delta_delta_mfcc = python_speech_features.delta(mfcc, 3)
    mfcc = np.hstack((np.hstack((mfcc, delta_mfcc)),delta_delta_mfcc))
    return mfcc

def readMFCC(base_dir, query_id, piece_type='reference', edit_type='n1', editTimeSec = 2):
    
    mfcc = None
    filename = None
    if(piece_type == 'queries'):
        filename = "{}/speech/queries/mfcc/160kbps/{}sec/{}_{}_160.pkl".format(base_dir, editTimeSec, query_id, edit_type)
    elif(piece_type == 'reference'):
        filename = "{}/speech/ref/mfcc/{}.pkl".format(base_dir,query_id)
        
    with open(filename, 'rb') as f:
        mfcc = pickle.load(f)
    f.close()
        
    return mfcc

def getPairwiseCostMatrix(queryFile, refFile):
    query, sr = librosa.load(queryFile, sr=16000)
    
    mfcc_query = python_speech_features.mfcc(query, sr, winstep = 0.01)
    delta_mfcc_query = python_speech_features.delta(mfcc_query, 2)
    delta_delta_mfcc_query = python_speech_features.delta(mfcc_query, 3)
    mfcc_query = np.hstack((np.hstack((mfcc_query, delta_mfcc_query)),delta_delta_mfcc_query))
    
    ref, sr = librosa.load(refFile, sr=16000)
    mfcc_ref = python_speech_features.mfcc(ref, sr, winstep= 0.01)
    delta_mfcc_ref = python_speech_features.delta(mfcc_ref, 2)
    delta_delta_mfcc_ref = python_speech_features.delta(mfcc_ref, 3)
    mfcc_ref = np.hstack((np.hstack((mfcc_ref, delta_mfcc_ref)),delta_delta_mfcc_ref))
    
    return euclidean_distances(mfcc_query, mfcc_ref)

#From 03
def cutMFCC(query_id, edit_type, piece):
    mfcc_ref = getMFCC(query_id)
    mfcc_query = getMFCC(query_id, piece_type="queries", edit_type=edit_type+str(piece))
    with open ('../ttemp/TamperingDetection/annots/160kbps_2sec.gt', 'r') as f:
        for row in f.readlines():
            if row.split(" ")[1] == '{}_{}{}'.format(query_id, edit_type, piece):
                start_time = float(row.split(" ")[2])
    total_time = librosa.get_duration(filename='../ttemp/TamperingDetection/speech/ref/{}.wav'.format(query_id))
    start_frame = int(start_time / total_time * mfcc_ref.shape[0])
    end_frame = int((start_time + 10)/ total_time * mfcc_ref.shape[0])
    mfcc_ref_cut = mfcc_ref[start_frame:] #TODO: FIGURE OUT IF end_frame NEEDS TO BE THE ENDING INDEX
    return mfcc_ref_cut, mfcc_ref, mfcc_query

#Identical, From 03
def getGtPath(query_id, ref_feat_size, query_feat_size, edit_type='n', piece=1, data_dir = '../ttemp/TamperingDetection'):
    filename = "{}_{}{}".format(query_id, edit_type, piece)
    with open("{}/annots/160kbps_2sec.gt".format(data_dir)) as f:
        lines = f.read().split('\n')
        for line in lines:
            testQuery = line.split(' ')[1]
            if testQuery[0:testQuery.rfind('_')] == filename:
                break
    tstart = float(line.split(' ')[2])
    tend = float(line.split(' ')[3])
    ref_len = librosa.get_duration(filename="{}/speech/ref/wav/{}.wav".format(data_dir, query_id))
    fstart = int(tstart / ref_len * ref_feat_size)
    fend = int(tend / ref_len * ref_feat_size)
    gt_path = []
    gt_label = []
    if edit_type == 'n':
        for i in range(query_feat_size):
            gt_path.append([i, fstart + i])
            gt_label.append('N')
    if edit_type == 'i':
        fins_pos = int(float(line.split(' ')[4]) / ref_len * ref_feat_size)
        fins_start = int(float(line.split(' ')[5]) / ref_len * ref_feat_size)
        fins_end = int(float(line.split(' ')[6]) / ref_len * ref_feat_size)
        for i in range(fins_pos):
            gt_path.append([i, fstart + i])
            gt_label.append('N')
        for i in range(fins_pos, fins_end - fins_start + fins_pos):
            gt_path.append([i, fins_pos + fstart])
            gt_label.append('I')
        for i in range(fins_end - fins_start + fins_pos, fend - fstart + fins_end - fins_start):
            gt_path.append([i, i + fstart - fins_end + fins_start])
            gt_label.append('N')
    if edit_type == 'd':
        fdel_start = int(float(line.split(' ')[4]) / ref_len * ref_feat_size)
        fdel_end = int(float(line.split(' ')[5]) / ref_len * ref_feat_size)
        for i in range(fdel_start):
            gt_path.append([i, fstart + i])
            gt_label.append('N')
        for i in range(fdel_start, fdel_end):
            gt_path.append([fdel_start, i + fstart])
            gt_label.append('D')
        for i in range(fdel_start, fend - fstart - fdel_end + fdel_start):
            gt_path.append([i, i + fstart + fdel_end - fdel_start])
            gt_label.append('N')
    if edit_type == 'r':
        frep_1_start = int(float(line.split(' ')[4]) / ref_len * ref_feat_size)
        frep_1_end = int(float(line.split(' ')[5]) / ref_len * ref_feat_size)
        frep_2_start = int(float(line.split(' ')[6]) / ref_len * ref_feat_size)
        frep_2_end = int(float(line.split(' ')[7]) / ref_len * ref_feat_size)
        for i in range(frep_1_start):
            gt_path.append([i, fstart + i])
            gt_label.append('N')
        for i in range(frep_1_start, frep_1_end):
            gt_path.append([i, i + fstart])
            gt_label.append('R')
        for i in range(frep_1_end, fend - fstart):
            gt_path.append([i, i + fstart])
            gt_label.append('N')
    return fstart, fend, np.array(gt_path), gt_label

#From 03
def labelPath(path, wp, labels, threshold=5):
    dists = cdist(wp, path)
    wp_labels = []
    wp_colors = []
    rules = {'N': 'black', 'I': 'red', 'D': 'green', 'R': 'blue'}
    for i, p in enumerate(wp):
        min_dist = np.min(dists[i])
        min_idx = np.argmin(dists[i])
        if min_dist > threshold:
            wp_colors.append(rules['R'])
        else:
            wp_colors.append(rules[labels[min_idx]])
        if wp_colors[-1] == "black":
            wp_labels.append('N')
        else:
            wp_labels.append('Y')
    return wp_colors, wp_labels

# def readMFCC(query_id, edit_type='n', piece=1, mfcc_type='old'):
#     ref_file_dir = '../ttemp/TamperingDetection/data/mfcc/reference/{}.pkl'.format(query_id.replace("_160", ""))
#     query_file_dir = '../ttemp/TamperingDetection/data/mfcc/queries/160kbps/2sec/{}_{}{}.pkl'.format(query_id, edit_type, piece)
#     with open(ref_file_dir, 'rb') as f:
#         d = pickle.load(f)
#         ref_mfcc = d[mfcc_type]
#     with open(query_file_dir, 'rb') as f:
#         d = pickle.load(f)
#         query_mfcc = d[mfcc_type]
#     return ref_mfcc.T, query_mfcc.T