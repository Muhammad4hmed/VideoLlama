import json 
import glob,time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(recognized, ground_truth, norm=True, bg_class=["backgroundbackgroundbackgroundbackgroundbackgroundbackgroundbackground"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def compute_f1_score(tp, fp, fn):
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def f_score_overlap(recognized, ground_truth, overlap, bg_class=["backgroundbackgroundbackgroundbackgroundbackgroundbackgroundbackground"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return compute_f1_score(float(tp), float(fp), float(fn))

def get_scores(json_file):
    f1_scores = []
    f1_25_scores = []
    ac_scores = []
    acc_scores = []
    sp_scores = []
    eds = []

    # EDIT HERE
    gts_p = load_dict_from_file(json_file)
    error = 0
    for sample in gts_p:
        #try:
            #print(sample['truth'].split('\n\n')[-1].split('-')[1])
            max_frames = int(sample['truth'].split('\n\n')[-2].split(' - ')[1].split(' ')[0])
            pred = np.array(['backgroundbackgroundbackgroundbackgroundbackgroundbackgroundbackground'] * (max_frames - 1))
            lis = np.array(['backgroundbackgroundbackgroundbackgroundbackgroundbackgroundbackground'] * (max_frames - 1))
            for line in (sample['pred'].split('\n')):
                if not '-' in line or line.strip() == '':
                    continue
                try:
                    num1, num2 = line.split(' - ')
                    num2, action = num2.split(' ')[0], "_".join(num2.strip().split(' ')[1:]).replace('\n','')
                    num1, num2 = int(num1) - 1 , int(num2) - 1
                    pred[num1:num2] = action
                except Exception as e:
                    print(f"at line 120 : {e}")
                    time.sleep(10)
                    continue

            for line in (sample['truth'].split('\n\n')):
                if not '-' in line or line.strip() == '':
                    continue
                try:
                    num1, num2 = line.split(' - ')
                    num2, action = num2.split(' ')[0], "_".join(num2.strip().split(' ')[1:]).replace('\n','')
                    num1, num2 = int(num1) - 1 , int(num2) - 1
                    lis[num1:num2] = action
                except Exception as e: 
                    print(f"at line 133 : {e}")
                    time.sleep(10)
                    continue
            
            score = f1_score(lis, pred, average='weighted')
            print(f"Score = {score}")
            #time.sleep(5)
            print(f"Lis = {lis}")
            print(f"Pred = {pred}")
            #f1_25 = f_score_overlap(lis, pred, overlap = 0.25)
            acc = accuracy_score(lis, pred)
            # print('='*10,'\n',sample['name'])
            # print('f1 score:', score)
            # print('MOF:', acc)
            f1_scores.append(score)
            #f1_25_scores.append(f1_25)
            acc_scores.append(acc)


            actions = []
            un_actions = np.unique(lis)
            for action in un_actions:
                if action in pred:
                    actions.append(action)

            spear = spearmanr(lis, pred).statistic

            action_acc = (len(actions)/len(un_actions))

            ed = edit_score(pred, lis)
            eds.append(ed)
            # print('action accuracy:', action_acc)
            ac_scores.append(action_acc)

            # print('spearman corr:', spear)
            sp_scores.append(spear)
        # except:
        #     error += 1
            # print('Error on', sample['name'])

    f1_scores = np.mean(f1_scores)
    f1_25_scores = np.mean(f1_25_scores)
    ac_scores = np.mean(ac_scores)
    acc_scores = np.mean(acc_scores)
    sp_scores = np.mean(sp_scores)
    eds = np.mean(eds)
    print('='*10)
    print('mean f1:', f1_scores)
    print('mean f1@25:', f1_25_scores)
    print('mean MOF:', acc_scores)
    print('mean action acc:', ac_scores)
    print('mean spearman correlation:', sp_scores)
    print('Edit Distance:', eds)
    print('Total Errors: ', error)

if __name__ == "__main__":
    get_scores('Predictions/Predictions_Breakfast.json')