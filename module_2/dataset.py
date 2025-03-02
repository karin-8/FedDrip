"""The ChestXDataset"""

from torch.utils.data import Dataset
from torchvision import transforms

class ChestXDataset(Dataset):
    def __init__(self, imlist):
        super().__init__()
        self.imlist = imlist
        self.transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(256),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225], inplace=True)])
    
    def __len__(self):
        return len(self.imlist)
    
    def __getitem__(self, idx):
        img = self.imlist[idx]
        img = img.convert("RGB")
        return self.transforms(img)