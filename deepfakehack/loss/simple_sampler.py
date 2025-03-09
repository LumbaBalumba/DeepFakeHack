from typing import List
from torch.utils.data.sampler import Sampler

from oml.utils.misc import smart_sample

import numpy as np

from collections import Counter, defaultdict
from typing import Iterator, List, Union

import numpy as np

from oml.interfaces.samplers import IBatchSampler
from oml.utils.misc import smart_sample


class SimpleSampler(IBatchSampler):

    def __init__(self, labels: Union[List[int], np.ndarray], batch_size: int, n_instances: int):
        n_labels=batch_size
  
        unq_labels = set(labels)

        assert isinstance(n_labels, int) and isinstance(n_instances, int)
        assert (1 < n_labels <= len(unq_labels)) and (1 < n_instances)

        self._labels = np.array(labels)
        self.n_labels = n_labels
        self.n_instances = n_instances

        self._batch_size = self.n_labels * self.n_instances
        self._unq_labels = unq_labels

        labels = np.array(labels)

        lbl2idx = defaultdict(list)

        for idx, label in enumerate(labels):
            lbl2idx[label].append(idx)

        self.lbl2idx = dict(lbl2idx)

        self.pos_classes_num = len(unq_labels) // 2

        self._unq_labels = set(np.arange(self.pos_classes_num))

        self._batches_in_epoch = len(self._unq_labels) // self.n_labels


    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._batches_in_epoch

    def __iter__(self) -> Iterator[List[int]]:
        inds_epoch = []

        labels_rest = self._unq_labels.copy()

        for _ in range(len(self)):
            ids_batch = []

            labels_for_batch = set(
                np.random.choice(list(labels_rest), size=min(self.n_labels // 2, len(labels_rest)), replace=False)
            )
            labels_rest -= labels_for_batch

            pos_labels = np.array(list(labels_for_batch)) * 2
            neg_labels = np.array(list(labels_for_batch)) * 2 + 1
            for cls in np.concatenate([pos_labels, neg_labels]):
                try:
                    cls_ids = self.lbl2idx[cls]
                    selected_inds = smart_sample(cls_ids, self.n_instances)
                    ids_batch.extend(selected_inds)
                except:
                    pass

            inds_epoch.append(ids_batch)

        return iter(inds_epoch)  # type: ignore
