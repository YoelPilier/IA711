# %%
from datasets import load_dataset

dataset = load_dataset("ylecun/mnist")
dataset

# %%

train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset = train_dataset.filter(
    lambda x: [l in [0, 1] for l in x["label"]], batched=True
)
test_dataset = test_dataset.filter(lambda x: x["label"] in [0, 1])


print(train_dataset[0]["label"])
train_dataset[0]["image"]

# %%
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        if self.transform:
            image = self.transform(image)
        label = self.dataset[idx]["label"]
        return image, label


train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_data = MNISTDataset(train_dataset, transform=train_transform)
test_data = MNISTDataset(test_dataset, transform=test_transform)

train_data[0][0].max(), train_data[0][0].min()

# %%
