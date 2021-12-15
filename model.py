import pprint

import torch
import torch.nn as nn
import numpy as np
from skimage import color, transform
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, classes=529):
        super().__init__()

        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64)
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.model8up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        )

        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.model9up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        )

        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.model10up = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
        )

        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=.2),
        )

        self.model3short8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.model2short9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.model1short10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.model_class = nn.Sequential(
            nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
        )

        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh()
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear')
        )

        self.softmax = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm

    def forward(self, input_A, input_B=None, mask_B=None):
        if (input_B is None):
            input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        if (mask_B is None):
            mask_B = input_A * 0

        input_tot = torch.cat((self.normalize_l(input_A), self.normalize_ab(input_B), mask_B), dim=1)
        model1 = self.model1(input_tot)
        model2 = self.model2(model1[:, :, ::2, ::2])
        model3 = self.model3(model2[:, :, ::2, ::2])
        model4 = self.model4(model3[:, :, ::2, ::2])
        model5 = self.model5(model4)
        model6 = self.model6(model5)
        model7 = self.model7(model6)
        model8up = self.model8up(model7) + self.model3short8(model3)
        model8 = self.model8(model8up)
        model9up = self.model9up(model8) + self.model2short9(model2)
        model9 = self.model9(model9up)
        model10up = self.model10up(model9) + self.model1short10(model1)
        model10 = self.model10(model10up)
        out = self.model_out(model10)

        return self.unnormalize_ab(out)


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    img_rgb_rs = transform.resize(img_rgb_orig, HW)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return tens_orig_l, tens_rs_l


def preprocess_img_ab(img_rgb_orig, HW=(256, 256), resample=3):
    img_rgb_rs = transform.resize(img_rgb_orig, HW)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 1: 3]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
    tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

    return tens_orig_l, tens_rs_l


def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    if (HW_orig[0] != HW[0] or HW_orig[1] != HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(np.asarray(out_lab_orig.data)[0, ...].transpose((1, 2, 0)))


def load_model(pretrained=True):
    model = Model()
    if pretrained:
        # pprint.pprint(torch.load("model.pth"))
        model.load_state_dict(torch.load("model.pth"))
    return model
