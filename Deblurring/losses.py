import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss




class PerceptualLoss():

    def initialize(self):
        self.activation = {}

        PATH = '/segmentation_pretrained_models/'  ## Deeplabv3plus pretrained model
        self.model = torch.load(PATH)
        print(self.model)
        with torch.no_grad():
            # self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



    def contentFunc(self):

        return self.model

    def get_activation(self,name):
        def hook(model, input, output):
            for nm in name:
                self.activation[nm] = output
        return hook

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0    ## retored Image
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])

        self.model.module.classifier.aspp.convs[3][2].register_forward_hook(self.get_activation(['low_feature_fake']))
        self.model.module.classifier.aspp.convs[3][2].register_forward_hook(self.get_activation(['high_feature_fake']))


        self.contentFunc.eval()
        with torch.no_grad():
            f_fake = self.contentFunc.forward(fakeIm)


        low_level_feature = self.activation['low_feature_fake']
        high_level_feature = self.activation['high_feature_fake']
        high_level_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                           align_corners=False)
        fake_fusion_layer = torch.cat([low_level_feature, high_level_feature ], dim=1)


        self.model.module.classifier.aspp.convs[3][2].register_forward_hook(self.get_activation(['low_feature_live']))
        self.model.module.classifier.aspp.convs[3][2].register_forward_hook(self.get_activation(['high_feature_live']))


        self.contentFunc.eval()
        with torch.no_grad():
            f_real = self.contentFunc.forward(realIm)

        low_level_feature = self.activation['low_feature_live']
        high_level_feature = self.activation['high_feature_live']
        high_level_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        live_fusion_layer = torch.cat([low_level_feature, high_level_feature], dim=1)


        f_real_no_grad = live_fusion_layer.detach()
        loss_perceptual = nn.MSELoss()(fake_fusion_layer, f_real_no_grad)  ## 이전 수정한것 ( conccat 된 부분 )


        return 0.05 * torch.mean(loss_perceptual)


    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

