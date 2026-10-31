import spaces
import torch
from model_loader import model, transforms, example_list
import gradio as gr
import PIL
import numpy as np

# place the model to cuda
model.to('cuda')

# function to perform inference
@spaces.GPU
def generate(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
  """Performs GPU-accelerated forward pass inference on an input image tensor.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded U-Net segmentation model.
    img : torch.Tensor
        A preprocessed 4D image tensor of shape (1, C, H, W) on the target device.

    Returns
    -------
    torch.Tensor
        Raw output logits tensor from the model.
  """
  # This function runs in the ZeroGPU worker (which is CUDA)
  img = img.to('cuda')

  with torch.inference_mode():
    logits = model(img)

  # Return a CPU tensor across the worker boundary
  return logits.cpu()

# function to run once submit is clicked
def on_click(img: PIL.Image.Image | None) -> np.ndarray | None:
  """Handles the UI submit event by preprocessing the image, running inference, and generating a binary mask.

    Parameters
    ----------
    img : PIL.Image.Image or None
        Input RGB image provided by the user via the UI.

    Returns
    -------
    np.ndarray or None
        A 2D uint8 NumPy array of shape (H, W) with pixel values 0 or 255 representing
        the predicted crack mask, or None if no image was provided.
  """
  if img:
    # Transform
    img = transforms(img).unsqueeze(0)

    # GPU inference happens inside the decorated function
    logits = generate(model, img)

    # predicted mask:
    mask = (logits.sigmoid() > 0.5).to(torch.int64)

    # Tensor -> NumPy
    mask = mask.squeeze((0, 1)).numpy()
    # 0/1 -> 0/255
    mask = (mask * 255).astype("uint8")

    return mask

  else:
    gr.Info(
        message="Upload an Image or choose from the samples",
        duration=4
    )
    return None

custom_css = """
  #txt {
    text-align: center;
  }

  #submit_btn {
    background: #FADA5E !important;
    font-weight: normal !important;
    color: #000000 !important;
  }

  #clear_btn {
    font-weight: normal !important;
  }

"""

with gr.Blocks() as demo:
  # title and message
  gr.Markdown(
    """
      ## U-NET Crack Segmentation
      > _This Web App runs a U-Net model to perform crack segmentation. **Black** pixels indicate no crack. **White** pixels show where the cracks appear_


    """, elem_id='txt')

  # input & output image row
  with gr.Row():
    input_img = gr.Image(
        type='pil', label='input', sources='upload',
        height=224, width=224, interactive=True, scale=1)

    output_img = gr.Image(
        label='mask', interactive=False, height=224, width=224, scale=1)

  # button row
  with gr.Row():
    with gr.Row(scale=1):
      submit_btn = gr.Button('✔️ Submit', variant='primary', size='md', elem_id='submit_btn')
      gr.ClearButton([input_img, output_img], value='🗑️ Clear', size='md', elem_id='clear_btn')
    with gr.Column(scale=1):
      pass

  # image examples
  with gr.Row():
    gr.Examples(
        fn=on_click,
        run_on_click=True,
        examples=example_list,
        inputs=input_img,
        outputs=output_img,
        label='Try these samples:',
        cache_examples=True,
        cache_mode='lazy',
        preload=False,
    )

    with gr.Column(scale=2):
      pass

  # footer
  gr.Markdown(
      """
      _Find GitHub repository and documentation linked [here](https://github.com/Martinmbiro/unet-crack-segmentation)_<br>
      """,
      elem_id='txt'
  )

  submit_btn.click(
      fn=on_click,
      inputs= input_img,
      outputs= output_img
  )

demo.queue().launch(css=custom_css)
