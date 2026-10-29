import torch
import os
from pathlib import Path
import torchvision.transforms.v2 as T
from huggingface_hub import hf_hub_download
import segmentation_models_pytorch as smp

# get list of example photos
BASE_PATH = Path(__file__).resolve().parent
example_list = list(BASE_PATH.joinpath('examples').rglob('*.jpg'))

transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(dtype=torch.float32, scale=True),
    T.Resize((224,224))
])

def load_model() -> torch.nn.Module:
  """Downloads fine-tuned weights from Hugging Face Hub and initializes the segmentation model.

    Returns
    -------
    torch.nn.Module
        A Segmentation Models PyTorch (smp) U-Net model with a ResNet-18 backbone,
        loaded with fine-tuned weights and set to evaluation mode on the target device.
  """
  # download weights
  weights_path = hf_hub_download(
    repo_id='martinmbiro/resnet18-unet-crack-segmentation',
    filename='resnet18-unet.pth',
    token=os.environ.get('HF_TOKEN')
  )

  # load weights
  state_dict = torch.load(f=weights_path, map_location='cpu')

  # initialize unet with resnet18 encoder backbone
  model = smp.Unet(
      encoder_name='resnet18',    # resnet18 backbone
      encoder_weights='imagenet', # with pre-trained weights
      in_channels=3,              # for RGB images
      classes=1                   # 1 output channel for crack probability
  )

  # load_state_dict, move model to device, set model to eval mode
  model.load_state_dict(state_dict)
  model.eval()

  # ZeroGPU-compatible module-level placement
  model.to("cuda")

  return model

model = load_model()
