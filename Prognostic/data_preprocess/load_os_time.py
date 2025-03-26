import pandas as pd
import json
import  os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bisect import bisect_left
from utils.block_utils import get_split_list

def load_tcga(tcga_csvfile, tcga_jsonfile):
    csv_data = pd.read_csv(tcga_csvfile)
    with open(tcga_jsonfile, 'r', encoding='utf-8') as load_f:
        json_data = json.load(load_f)

    for ijson_data in json_data:
        sample_name = ijson_data['name']
        csv_os_time = csv_data[csv_data['Patient ID']==sample_name]["DFS"].values[0]
        ijson_data['OS'] = csv_os_time

    with open("E:\\data\\hs_cyhz_youan_tcga_fudan\\rfs_new_tcga.json", "w", encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    return json_data

def load_self_data(hisp_name, jsonfile):
    with open(jsonfile, 'r', encoding='utf-8') as load_f:
        json_data = json.load(load_f)
    if "cyhz_all_os" in jsonfile:
        hzsub_json_data = []
        cysub_json_data = []
        for ijson_data in json_data:
            ijson_data['OS'] = float(ijson_data['OS']) / 30.
            ijson_data['path'] = ijson_data['path'].replace("/HZ/", "/")
            if "hz-info" in ijson_data['path'] or "_hz2" in ijson_data['path']:
                hzsub_json_data.append(ijson_data)
            else:
                cysub_json_data.append(ijson_data)
        with open(f"E:\\data\\hs_cyhz_youan_tcga_fudan\\rfs_new_hz.json", "w", encoding='utf-8') as f:
            json.dump(hzsub_json_data, f, ensure_ascii=False, indent=4)
        with open(f"E:\\data\\hs_cyhz_youan_tcga_fudan\\rfs_new_cy.json", "w", encoding='utf-8') as f:
            json.dump(cysub_json_data, f, ensure_ascii=False, indent=4)
        return json_data
    else:
        for ijson_data in json_data:
            ijson_data['OS'] = float(ijson_data['OS'])/ 30.
        with open(f"E:\\data\\hs_cyhz_youan_tcga_fudan\\rfs_new_{hisp_name}.json", "w", encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        return json_data


def max_min_transform(OS_time, start, end):
    max_os_time = float(end)
    min_os_time = float(start)
    scale_os_time = (OS_time - min_os_time) / (max_os_time - min_os_time)
    return scale_os_time

def scaled_sigmoid(x, start, end):
    """
    when x in start and end, the value should be in [0, 1]
    """
    # b = np.abs(start - end)
    b = 100
    score = 2/(1+np.exp(-np.log(40000)*(x-b)/b+np.log(5e-3)))
    return score / 2

# def merge_hist(hist, bin_edges, counts=4):
#     process_bin_edges = bin_edges[1:]
#     hist_num = len(hist)
#     for idx in range(1, hist_num-1):
#         if hist[idx] < counts:
#             left_hist_num = hist[idx-1]
#             right_hist_num = hist[idx+1]
#             if left_hist_num < right_hist_num:



if __name__ == '__main__':
    # new_json_data = load_tcga(
    #     "E:\\data\\hs_cyhz_youan_tcga_fudan\\tcga.csv",
    #     "E:\\data\\hs_cyhz_youan_tcga_fudan\\tcga_all_rfs.json"
    # )
    # new_json_data = load_self_data(
    #     "youan",
    #     "E:\\data\\hs_cyhz_youan_tcga_fudan\\youan_rfs.json"
    # )
    # new_json_data = load_self_data(
    #     "hs",
    #     "E:\\data\\hs_cyhz_youan_tcga_fudan\\hs_all_rfs.json"
    # )
    # new_json_data = load_self_data(
    #     "cyhz",
    #     "E:\\data\\hs_cyhz_youan_tcga_fudan\\cyhz_all_os.json"
    # )


    all_files = ['rfs_check_cy.json', 'rfs_check_huashan.json', 'rfs_check_huashan2.json', 'rfs_check_hz.json', 'rfs_check_youan.json']
    # all_files = ['rfs_new_hs.json']
    all_OS_time = []
    event_OS_time = []
    no_event_OS_time = []



    for ifile in all_files:
        hisp_samples = []
        with open(os.path.join("/home/whisper/Disk2/data/hs_cyhz_youan_tcga_fudan/SurviveData", ifile), 'r', encoding='utf-8') as load_f:
            json_data = json.load(load_f)
        for isample in json_data:
            all_OS_time.append(isample['RFS_time']/30.)
            if int(isample['events']) == 0:
                no_event_OS_time.append(isample['RFS_time']/30.)
            elif int(isample['events']) == 1:
                event_OS_time.append(isample['RFS_time']/30.)
            else:
                raise "unknown event"

            hisp_samples.append(isample["PID"])
        print(f"{ifile} with samples {len(set(hisp_samples))}.")
        '''
        rfs_check_cy.json with samples 352.
        rfs_check_huashan.json with samples 668.
        rfs_check_huashan2.json with samples 531.
        rfs_check_hz.json with samples 159.
        rfs_check_youan.json with samples 298.
        '''
    max_os = 100
    min_os = 0
    split_num = 500
    # max_os = max(all_OS_time)
    # min_os = min(all_OS_time)
    print(max_os, min_os, len(all_OS_time))
    ultra_split_num = 0
    ultra_start = 9
    ultra_end = 50
    split_list = get_split_list(split_num, ultra_split_num, ultra_start, ultra_end)
    hist, bin_edges = np.histogram(
        # [scaled_sigmoid(i, min_os, max_os) for i in no_event_OS_time],
        [max_min_transform(i, min_os, max_os) for i in all_OS_time],
        # all_OS_time,
        bins=split_list
    )
    print(hist)
    print(sum(hist))
    print(bin_edges)


    # print(bisect_left(split_area, 0.0001))

    # pd_os_time = pd.DataFrame(np.array(all_OS_time))
    # pd_os_time.plot(kind='hist', bins=10)
    # plt.show()
    #

    #
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    # y = sorted(all_OS_time)
    # print(scaled_sigmoid(110, min_os, max_os))
    # print(max_min_transform(110, min_os, max_os))
    # ax1.scatter(y, [scaled_sigmoid(i, min_os, max_os) for i in y], s=1)
    # ax1.set_title("fit (scaled_sigmoid)")
    #
    # ax2.scatter(y, [max_min_transform(i, min_os, max_os) for i in y], s=1)
    # ax2.set_title("fit (max-min)")
    #
    # plt.show()