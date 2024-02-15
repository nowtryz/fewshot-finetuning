import numpy as np
from monai.data import Dataset
from monai.transforms import apply_transform


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = [img["name"].split("/")[0] for img in data]
        self.datasetkey = list(np.unique(keys))

        self.data_dic = {
            iKey: [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
            for iKey in self.datasetkey
        }

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        # the index generated outside is only used to select the dataset
        # the corresponding data in each dataset is selected by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]

        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)
