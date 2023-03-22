# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add class stats computation

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image

#在mmcv.track_parallel_progress处调用这个函数，每次传入一个json_file(str)
def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png') #e.g aachen_000000_000019_gtFine_polygons.json -> aachen_000000_000019__labelTrainIds.png

    '''官方接口。Reads labels as polygons in JSON format and converts them to label images,
    where each pixel has an ID that represents the ground truth label.
    json2labelImg(json_file(输入的json), label_file(输出的label image), 'trainIds'(只保留需要训练的类别，好像是19类))
    '''
    json2labelImg(json_file, label_file, 'trainIds')

    if 'train/' in json_file:
        pil_label = Image.open(label_file)
        label = np.asarray(pil_label)
        sample_class_stats = {}
        for c in range(19): #19个类别
            n = int(np.sum(label == c)) #统计每个类别的数量
            if n > 0:
                sample_class_stats[int(c)] = n #记录数量
        sample_class_stats['file'] = label_file
        return sample_class_stats #用于RSC
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
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
    cityscapes_path = args.cityscapes_path #e.g path/to/cityscape
    out_dir = args.out_dir if args.out_dir else cityscapes_path #如果out_dir没有指定的话，就传到path/to/cityscape
    mmcv.mkdir_or_exist(out_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir) #e.g path/to/cityscape/gtFine
    
    '''mmcv.scandir(dir_path(数据路径),suffix(需要被选的文件路径),recursive(递归地scan这个目录,
       比如此目录下有train,val,test,这三个目录下又分别有aachen, bremen, darmstadt, erfurt等.
       会递归地依次进入这些目录))'''
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly) #e.g data/to/gt/aachen/aachen_000000_000019_gtFine_polygons.json
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

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
