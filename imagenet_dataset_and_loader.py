# https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from torchvision import transforms


class ImageNetKaggle(Dataset):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def __init__(self, root, split, transform=val_transform):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
        self.types = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNetValTestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, num_workers=4, loader_kwargs={}):
        if not isinstance(dataset, ImageNetKaggle):
            raise ValueError("dataset only be type `ImageNetKaggle`")
        if not (dataset.types == "val" or dataset.types == "test"):
            raise ValueError("dataset only be type `val` or `test`")
        # overwrite anyway
        loader_kwargs["shuffle"] = False
        loader_kwargs["drop_last"] = False
        loader_kwargs["pin_memory"] = True
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, **loader_kwargs)


if __name__ == "__main__":
    dataset = ImageNetKaggle("D:\WORKINGSPACE\Corpus\ImageNet", "val")
    dataloader = ImageNetValTestDataLoader(dataset)
    import tqdm

    total = 0
    for x, y in tqdm.tqdm(dataloader):
        total += len(y)
    print(f"total={total}")
