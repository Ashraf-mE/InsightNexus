import torch
import torch.nn as nn
from torchvision import models

def efficientnet():
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    old_fc = model.classifier[-1]
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features=7, bias=True)
    model.classifier[-1] = new_fc
    return model

def load_finetuned_model(model_path):
    model = efficientnet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model