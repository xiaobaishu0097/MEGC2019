import torch

from PIL import Image
from torch import stack
from torch.utils.data import Dataset

class Flow_loader(Dataset):
    def __init__(self, data_dir, label_dir, domain_label_dir, samples, transform=None, label_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.domain_label_dir = domain_label_dir

        with open(self.data_dir, 'r') as d:
            self.data = d.readlines()

        with open(self.label_dir, 'r') as l:
            self.label = l.readlines()

        with open(self.domain_label_dir, 'r') as dl:
            self.domain_label = dl.readlines()

        self.transform = transform
        self.label_transform = label_transform

        self.samples = samples
        self.size = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        flow_sample = self.samples[index]

        flow_path = '.' + self.data[flow_sample].strip('\n')

        flow = Image.open(flow_path).convert('RGB')

        file_name = flow_path.split("/")[-2:-1]

        label = int(float(self.label[sample].strip('\n')))

        domain_label = 0 if self.domain_label[sample] == 'Macro\n' else 1

        if self.transform:
            flow = self.transform(flow)

        if self.label_transform:
            label = self.label_transform(label)

        return {"image":  flow, "label": label, "domain_label": domain_label, "file_name": file_name}

    def __len__(self):
        return self.size