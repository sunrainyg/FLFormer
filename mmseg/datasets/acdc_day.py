from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ACDCDayDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(ACDCDayDataset, self).__init__(
            img_suffix='_rgb_ref_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            split=None,
            **kwargs)
