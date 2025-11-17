import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
import random
import json
import pandas as pd

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels_dic: dict, steps: int, batch_size: int, grid_size=9, datset_len=0, cfg=None, way="train"):
        self.one_cls_flag = False
        self.cfg = cfg
        self.way = way
        self.patch_pixels = cfg["dataset"]["patch_size"] ** 2
        if num_classes == 1:
            self.num_classes = num_classes + 1
            self.one_cls_flag = True
        else:
            self.num_classes = num_classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))#初始化混淆矩阵，元素都为0
        self.cal_pixel = cfg["val"].get("cal_pixel", False)
        self.matrix_pixels = np.zeros((self.num_classes, self.num_classes))#初始化混淆矩阵，元素都为0
        self.labels = [label for cls_name, label in labels_dic.items()]#类别标签
        self.labels_dic = {}
        self.all_lables = np.array([])
        self.all_preds = np.array([])
        self.all_preds_logit = np.array([])
        self.steps = steps
        self.batch_size = batch_size
        self.patch_nums = grid_size

        self.flag = 0
        self.thre = 0.5 if way=="train" else cfg["val"]["thresh"]
        for k, v in labels_dic.items():
            self.labels_dic[v] = k

    def fast_matrix(self, a , b, matrix, n=2):
        # 寻找GT中为目标的像素索引
        k = (a >= 0) & (a < n)
        # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
        inds = n * a[k].astype(int) + b[k]
        matrix += np.bincount(inds.astype(int), minlength=n**2).reshape(n, n)
        return matrix


    def update(self, preds_logit, labels, pixel_label_flat = None):
        self.all_lables = np.append(self.all_lables, labels)
        if self.one_cls_flag:
            preds_logit = preds_logit.sigmoid()
            preds = torch.zeros_like(torch.tensor(labels))
            preds[preds_logit > self.thre] = 1
        else:
            preds = torch.max(preds_logit, dim=1)[1]
        preds = preds.numpy()
        self.all_preds = np.append(self.all_preds, preds)
        self.all_preds_logit = np.append(self.all_preds_logit, preds_logit)

        # if preds.sum() < preds.shape[0] // 1.2:
        #     precision, recall, thresholds = precision_recall_curve(preds, labels)
        #     aupr = auc(recall, precision)
        #     print(aupr)
            # Con = np.bincount((2 * labels[(labels >= 0) & (labels < 2)].astype(int) + preds[(labels >= 0) & (labels < 2)]).astype(int), minlength=2**2).reshape(2, 2)
            # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # for p, t in zip(preds, labels):
        #     self.matrix[int(p), int(t)] += 1 # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1
        self.matrix = self.fast_matrix(labels, preds, self.matrix)
        # preds_logit = preds_logit.numpy() # b, 4
        # for i in range(self.num_classes):
        #     start = self.flag * self.batch_size * self.patch_nums
        #     end = start + preds_logit.shape[0]
        #     if self.one_cls_flag:
        #         if i == 0:
        #             self.all_preds[i][start:end] = 1 - preds_logit
        #         else:
        #             self.all_preds[i][start:end] = preds_logit
        #     else:
        #         self.all_preds[i][start:end] = preds_logit[:, i]

        self.flag += 1
        
        ############ calculate pixels-level matrix #############


        # print("preds",preds.shape, type(preds), preds)
        # print("pixel_label_flat",pixel_label_flat.shape, type(pixel_label_flat))
        # for p,t in zip(preds, pixel_label_flat):
        # for p,t in zip(pixel_label_flat, preds):
        #     self.matrix_pixels[int(p), int(t)] += 1
        if self.cal_pixel:        
            preds = np.array([np.ones(self.patch_pixels)*i for i in preds]).flatten()
            pixel_label_flat = pixel_label_flat.flatten()
            # print(pixel_label_flat.shape, preds.shape)
            self.matrix_pixels = self.fast_matrix(pixel_label_flat, preds, self.matrix_pixels)
        
        ############ end #############


    def summary(self, vis=True):#计算指标函数
        summery_dic = {}
        # calculate accuracy
        sum_TP = 0
        self.matrix = self.matrix.T
        self.matrix_pixels = self.matrix_pixels.T
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n #总体准确率
        print("the model accuracy is ", acc)
		
        cls_IoU = np.diag(self.matrix) / (self.matrix.sum(1) + self.matrix.sum(0) - np.diag(self.matrix))
        cls_Dice = 2*cls_IoU / (1+cls_IoU)
        cls_IoU_pixels = np.diag(self.matrix_pixels) / (self.matrix_pixels.sum(1) + self.matrix_pixels.sum(0) - np.diag(self.matrix_pixels))
        cls_Dice_pixels = 2*cls_IoU_pixels / (1+cls_IoU_pixels)

		# kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 4)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        self.table = PrettyTable()#创建一个表格
        self.table.field_names = ["", "Precision", "Recall", "Specificity", "F1", "cls_acc", "IoU","IoU_p", "Dice", "Dice_p", "acc", "mIoU", "mIoU_p", "mDice", "mDice_p", "kappa"]
        # self.table.field_names = ["", "Precision", "Recall", "Specificity", "F1", "AUC", "cls_acc", "IoU","IoU_p", "Dice", "Dice_p", "acc", "mIoU", "mIoU_p", "mDice", "mDice_p", "kappa"]
        for i in range(self.num_classes):#精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0. #每一类准确度
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall + 1e-8), 4)
            cls_acc = round((TP + TN) / (TN + FP + TP + FN), 4)
            # tmp_labels = (self.all_lables == i) # convert to one-hot
            # AUC = round(roc_auc_score(tmp_labels, self.all_preds[i]), 4)

            IoU = round(cls_IoU[i], 4)
            Dice = round(cls_Dice[i], 4)
            IoU_pixels = round(cls_IoU_pixels[i], 4)
            Dice_pixels = round(cls_Dice_pixels[i], 4)
            self.table.add_row([self.labels_dic[i], Precision, Recall, Specificity, F1, cls_acc, IoU, IoU_pixels, Dice, Dice_pixels, "", "", "", "", "", ""])
            # self.table.add_row([self.labels_dic[i], Precision, Recall, Specificity, F1, AUC, cls_acc, IoU, IoU_pixels, Dice, Dice_pixels, "", "", "", "", "", ""])
        mIoU = round(np.sum(cls_IoU)/len(cls_IoU), 4)
        mDice = round(np.sum(cls_Dice)/len(cls_Dice), 4)
        mIoU_p = round(np.sum(cls_IoU_pixels)/len(cls_IoU_pixels), 4)
        mDice_p = round(np.sum(cls_Dice_pixels)/len(cls_Dice_pixels), 4)
        self.table.add_row(["", "", "", "", "", "", "", "", "", "",round(acc, 4), round(mIoU, 4), round(mIoU_p, 4), round(mDice, 4), round(mDice_p, 4), kappa])
        # self.table.add_row(["", "", "", "", "", "", "", "", "", "", "",round(acc, 4), round(mIoU, 4), round(mIoU_p, 4), round(mDice, 4), round(mDice_p, 4), kappa])
        if vis:
            print(self.table)

        F1_lst, Precision_lst, Recall_lst, Specificity_lst, IoU_p_lst, Dice_p_lst  = [], [], [], [], [], []
        for idx, row in enumerate(self.table):
            row.border = False
            row.header = False
            F1_lst.append(float(row.get_string(fields=["F1"]).strip()))
            Precision_lst.append(float(row.get_string(fields=["Precision"]).strip()))
            Recall_lst.append(float(row.get_string(fields=["Recall"]).strip()))
            Specificity_lst.append(float(row.get_string(fields=["Specificity"]).strip()))
            IoU_p_lst.append(float(row.get_string(fields=["IoU_p"]).strip()))
            Dice_p_lst.append(float(row.get_string(fields=["Dice_p"]).strip()))
            if idx == self.num_classes - 1:
                break


        precision, recall, thresholds = precision_recall_curve(self.all_lables, self.all_preds_logit)
        aupr = auc(recall, precision)
        print(f"\n**********************\n sklearn \n**********************\n ")
        cm = confusion_matrix(self.all_lables, self.all_preds)
        print(cm)

        cm = confusion_matrix(self.all_lables, self.all_preds, normalize="true")
        print("混淆矩阵:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()

        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        acc = accuracy_score(self.all_lables, self.all_preds)
        print(f"sklearn acc: {acc}")
        p = precision_score(self.all_lables, self.all_preds)
        print(f"sklearn precision: {p}")
        r = recall_score(self.all_lables, self.all_preds)
        print(f"sklearn recall: {r}")
        f1score = f1_score(self.all_lables, self.all_preds)
        print(f"sklearn f1score: {f1score}")
        # auc_ROCscore = roc_auc_score(self.all_lables, self.all_preds)
        # print(f"sklearn auc-roc: {auc_ROCscore}")

        self.plot_aucroc(self.all_lables, self.all_preds_logit)
        self.plot_aucpr(self.all_lables, self.all_preds_logit)
        print(f"\n**********************\n AUC-PR:{aupr} \n**********************\n ")

        summery_dic["cancer_F1"] = F1_lst[1]
        summery_dic["cancer_Precision"] = Precision_lst[1]
        summery_dic["cancer_Recall"] = Recall_lst[1]
        summery_dic["cancer_Specificity"] = Specificity_lst[1]
        summery_dic["cancer_IoU_p"] = IoU_p_lst[1]
        summery_dic["cancer_Dice_p"] = Dice_p_lst[1]

        summery_dic["F1"] = sum(F1_lst)/len(F1_lst)
        summery_dic["Precision"] = sum(Precision_lst)/len(Precision_lst)
        summery_dic["Recall"] = sum(Recall_lst)/len(Recall_lst)
        summery_dic["Specificity"] = sum(Specificity_lst)/len(Specificity_lst)
        summery_dic["acc"] = acc
        summery_dic["mIoU"] = mIoU
        summery_dic["mDice"] = mDice
        summery_dic["mIoU_p"] = mIoU_p
        summery_dic["mDice_p"] = mDice_p



        return summery_dic

    def plot(self):#绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

    def plot_aucroc(self, y_test, y_pred_proba):

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        pd_data = {
            "fpr": fpr,
            "tpr": tpr,
        }
        pd.DataFrame(pd_data).to_csv("auc_ROC.csv")
        auc_ROCscore = roc_auc_score(y_test, y_pred_proba)
        print(f"auc_ROCscore: {auc_ROCscore}")
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue',
                 label=f'AUC = {auc_ROCscore}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random Guess')

        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right')
        # plt.grid(True, alpha=0.3)

        # 添加对角线参考线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # 显示图形
        plt.tight_layout()


        plt.savefig("cls_auc_roc.jpg",dpi=300)

    def plot_aucpr(self, y_test, y_pred_proba):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        aupr = auc(recall, precision)

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AUC PR: {aupr:.3f}')


        # 设置图形属性
        plt.xlim([0.0, 1.05])
        plt.ylim([0.5, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)

        plt.legend(loc='lower right')
        # plt.grid(True, alpha=0.3)
        # 显示图形
        plt.tight_layout()
        plt.savefig("cls_pr.jpg",dpi=300)

def load_pretrain_checkpoint(pth, net):
    print("Use pretrian ckpt: ", pth)
    assert len(pth) != 0, "Please input a pretrain valid ckpt_path"
    net_dict = net.state_dict()
    checkpoint = torch.load(pth)
    try:
        pretrained_dict = checkpoint['state_dict']
    except:
        pretrained_dict = checkpoint
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and ("fc" not in k) and ('crf' not in k)}
    missing_keys = pretrained_dict.keys() - net_dict.keys()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.keys() and "fc" not in k}
    excepted_key = net_dict.keys() - pretrained_dict.keys()
    print("current key nums: {} , use pretrain key nums: {}".format(len(net_dict.keys()), len(pretrained_dict.keys())))
    print("missing_keys : {}, excepted_key: {}".format(missing_keys, excepted_key))
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    return net

def load_tif_dict(data_path, mode, epoch_nums=0):
    data_file = open(data_path, "r", encoding="utf-8")
    data_lines = data_file.readlines()
    data_file.close()
    data_dict = json.loads(data_lines[0])

    data_length = len(data_dict)
    # 每个epoch训练sample_length个数据
    if data_length > 10:
        sample_length = data_length // 10
    else:
        sample_length = data_length
    # 每个数据至少被重复sample_num次
    sample_num = 10

    if mode == 'train':
        # 生成必须的采样列表
        required = [i for i in range(data_length) for _ in range(sample_num)]
        random.shuffle(required)

        samples = []
        for _ in range(epoch_nums):
            current_sample = []
            # 从required列表中取出尽可能多的元素
            take = min(sample_length, len(required))
            if take > 0:
                current_sample.extend(required[:take])
                del required[:take]

            # 补充剩余样本
            remaining = sample_length - take
            if remaining > 0:
                supplemental = random.choices(range(data_length), k=remaining)
                current_sample.extend(supplemental)

            current_dict = {}
            for idx, sample_name in enumerate(data_dict):
                if idx in current_sample:
                    current_dict[sample_name] = data_dict[sample_name]
            samples.append(current_dict)
        return samples
    else:
        return [data_dict]