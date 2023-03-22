# FLFormer

<a>
<img src="https://img.shields.io/badge/1st%20prize-CVPR2022%20Workshop%20Vision%20for%20All%20Seasons-yellowgreen" />
</a>

<a>
    <img src="https://img.shields.io/badge/unsupervised%20learning-domain%20adaptation%20segmentation-blue" />
</a>


## &#x1F680; Overview
Our target task is unsupervised domain adaptive segmentation, the dataset used is Cityscapes-to-ACDC, there are four domains in total (fog, snow, night, rain), each domain of ACDC has two files, for example, day and night images, they are roughly aligned one by one by GPS positioning.

*Overview of our method*:

## Setup Environment

We recommend setting up a new virtual environment:

```shell
conda create -n flformer python==3.8.5
```

In that environment, the requirements can be installed with:

```shell
conda activate flformer
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

If problems occur with the automatic download of pretained model weights, please follow the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```


## Inference Demo



## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

The final folder structure should look like this:

```none
FIFormer
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Training

For the experiments in our paper (e.g. network architecture comparison,
component ablations, ...), we use a system to automatically generate
and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

## Testing & Predictions

The provided  checkpoint trained on GTA->Cityscapes
(already downloaded by `tools/download_checkpoints.sh`) can be tested on the
Cityscapes validation set using:

```shell
sh test.sh work_dirs/211108_1622_gta2cs_flformer_s0_7f24c
```

The predictions are saved for inspection to
`work_dirs/211108_1622_gta2cs_flformer_s0_7f24c/preds`
and the mIoU of the model is printed to the console. The provided checkpoint
should achieve 68.85 mIoU. Refer to the end of
`work_dirs/211108_1622_gta2cs_flformer_s0_7f24c/20211108_164105.log` for
more information such as the class-wise IoU.

Similarly, also other models can be tested after the training has finished:

```shell
sh test.sh path/to/checkpoint_directory
```

When evaluating a model trained on Synthia->Cityscapes, please note that the 
evaluation script calculates the mIoU for all 19 Cityscapes classes. However, 
Synthia contains only labels for 16 of these classes. Therefore, it is a common
practice in UDA to report the mIoU for Synthia->Cityscapes only on these 16 
classes. As the Iou for the 3 missing classes is 0, you can do the conversion 
mIoU16 = mIoU19 * 19 / 16.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for FLFormer are:


## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [DAFormer]()
* [Bi-mix]()

