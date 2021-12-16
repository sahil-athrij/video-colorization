from os import listdir

import torch
import torch.optim as optim
from cv2 import cv2
from torch import nn

from torch.utils.data import DataLoader, Dataset
from model import load_model, preprocess_img, preprocess_img_ab
from network import Model


class MyDataset(Dataset):

    def __init__(self, root, transform_x=None, transform_y=None):
        self.transformX = transform_x
        self.transformY = transform_y

        x = []
        y = []

        for i in range(len(listdir(root))):
            image = cv2.imread(f"{root}/{i}.png")
            original, image_processed = preprocess_img(image)

            x.append(image_processed)
            y.append(preprocess_img_ab(image)[1])

        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        im = self.X[idx]
        label = torch.cat((self.Y[idx-1 if idx != 0 else 0], self.Y[idx]), dim=1)[0]
        im_prev = im if idx == 0 else self.X[idx - 1]

        if self.transformX is not None:
            im = self.transformX(im)

        if self.transformY is not None:
            label = self.transformY(label)

        return (im_prev, im), label


def train(pix, generator, batch_size=1):
    train_loader = torch.utils.data.DataLoader(MyDataset(root="input/x"), batch_size=batch_size, shuffle=True,
                                               num_workers=0, drop_last=True)

    generator_optimiser = optim.SGD(generator.parameters(), lr=0.03)
    generator_loss = nn.MSELoss()

    for epoch in range(5):
        for images, label in train_loader:
            img = images[0][0]
            prev = images[1][0]

            generator_optimiser.zero_grad()

            pix1 = pix(prev)
            pix2 = pix(img)

            output = generator(pix1, pix2)

            loss = generator_loss(output, label)

            loss.backward()
            generator_optimiser.step()


if __name__ == "__main__":
    train(load_model(True), Model())
