import torch
from optimum.bettertransformer import BetterTransformer

from transformers import CLIPModel

from torch.autograd import Variable
import copy

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
model.eval()

model_bt = BetterTransformer.transform(model, keep_original_model=True)

pixel_values = Variable(torch.randn(1, 3, 448//2, 448//2).cuda().float(), requires_grad=True)#.retain_grad()
pixel_values1 = copy.deepcopy(pixel_values)

y1 = model.get_image_features(pixel_values=pixel_values)


y2 = model_bt.get_image_features(pixel_values=pixel_values)

assert torch.allclose(y1, y2, atol=1e-3)
