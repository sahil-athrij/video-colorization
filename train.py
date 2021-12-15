from os import listdir

import torch
from cv2 import cv2

from torch.utils.data import DataLoader, Dataset
from model import load_model, preprocess_img, preprocess_img_ab, postprocess_tens
from network import Model


class MyDataset(Dataset):

    def __init__(self, root, transformX=None, transformY=None):
        self.transformX = transformX
        self.transformY = transformY

        x = []
        y = []

        for img in listdir(root):
            image = cv2.imread(f"{root}/{img}")
            original, image_processed = preprocess_img(image)

            x.append(image_processed)
            y.append(preprocess_img_ab(image)[1])

        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        im = self.X[idx]
        label = self.Y[idx]
        im_prev = im if idx == 0 else self.X[idx - 1]

        if self.transformX is not None:
            im = self.transformX(im)

        if self.transformY is not None:
            label = self.transformY(label)

        return (im_prev, im), label


def train(pix, generator, batch_size=8):
    train_loader = torch.utils.data.DataLoader(MyDataset(root="input/sunith"), batch_size=batch_size, shuffle=True,
                                              num_workers=2, drop_last=True)

    for epoch in range(5):
        for images, label in train_loader:
            pix1 = pix(images[0])
            pix2 = pix(images[1])

            output = generator(pix1, pix2)

            print(output.shape)


if __name__ == "__main__":
    train(load_model(True), Model())
