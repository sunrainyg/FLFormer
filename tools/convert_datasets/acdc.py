# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image

#在mmcv.track_parallel_progress处调用这个函数，每次传入一个label_file(png)
def convert_json_to_label(label_file):

    pil_label = Image.open(label_file)
    label = np.asarray(pil_label)
    sample_class_stats = {}
    for c in range(19): #19个类别
        n = int(np.sum(label == c)) #统计每个类别的数量
        if n > 0:
            sample_class_stats[int(c)] = n #记录数量
    sample_class_stats['file'] = label_file
    return sample_class_stats #用于RSC



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert acdc annotations to TrainIds')
    parser.add_argument('acdc_path', help='cityscapes data path') #/data3/yl/datasets/ACDC
    parser.add_argument('--gt-dir', default='gt/night/train', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    #获取数据和标签的路径，定义输出路径
    acdc_path = args.acdc_path #e.g path/to/acdc
    out_dir = args.out_dir if args.out_dir else acdc_path #如果out_dir没有指定的话，就传到path/to/acdc
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(acdc_path, args.gt_dir) #e.g path/to/acdc/gtFine
    
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_labelTrainIds.png', recursive=True):
        poly_file = osp.join(gt_dir, poly) #e.g data/to/gt/aachen/xxxx_labelTrainIds.png
        poly_files.append(poly_file) #把所有json文件名存入poly_files

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:#nproc: num workers
            #mmcv.track_parallel_progress 并行任务的跟踪进度 （func(对每个task作用的func), tasks(一系列的task), nproc(num_workers)）
            sample_class_stats = mmcv.track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
