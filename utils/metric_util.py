import numpy as np
import torch
import torch.distributed as dist
from mmengine.logging.logger import MMLogger

class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices) # 4类
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()
    
    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item() # true positive + false negative
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item() # true positive
            self.total_positive[i] += torch.sum(outputs == c).item() # true positive + false positive

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                #  true positive / (true positive + false negative) + (true positive + false positive) - true positive
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100 
        
class MeanIoU_test:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices # array([ 1,  2,  3,  4])
        self.num_classes = len(label_str) 
        self.ignore_label = ignore_label # 0
        self.label_str = label_str # ['clear', 'rain15', 'rain33', 'rain55', 'light', 'medium', 'heavy', 'foga', 'fogb', 'fogc']
        self.subcondition_to_index = {label: i for i, label in enumerate(self.label_str)}
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets, subcondition):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        subcondition_to_c = {
            'rain15': 2, 'rain33': 2, 'rain55': 2,  
            'light': 4, 'medium': 4, 'heavy': 4, 
            'foga': 3, 'fogb': 3, 'fogc': 3  
        }
        if subcondition not in subcondition_to_c:
            print(f"Unknown subcondition: {subcondition}")
            return

        c = subcondition_to_c[subcondition]  

        index = self.subcondition_to_index[subcondition]  
        self.total_seen[index] += torch.sum(targets == c).item()  
        self.total_correct[index] += torch.sum((targets == c) & (outputs == c)).item()  
        self.total_positive[index] += torch.sum(outputs == c).item()  

        self.total_seen[0] += torch.sum(targets == 1).item()
        self.total_correct[0] += torch.sum((targets == 1)
                                               & (outputs == 1)).item()
        self.total_positive[0] += torch.sum(outputs == 1).item()

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):

            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                #  true positive / (true positive + false negative) + (true positive + false positive) - true positive
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = MMLogger.get_current_instance()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100 