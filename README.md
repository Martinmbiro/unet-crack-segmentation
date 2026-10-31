# Crack Segmentation with ResNet-18 U-Net

Hello again 👋

+ This project entails training, evaluating and deploying a [U-Net](https://arxiv.org/abs/1505.04597) model with a pre-trained [ResNet-18](https://arxiv.org/html/2510.24036v1) encoder backbone. The purpose of the model is to **segment cracks** on structures. This idea could be extended to defect identification and quality assurance.
+ Find the interactive web-base user interface on my Hugging Face Space [linked here](https://huggingface.co/spaces/martinmbiro/crack-segmentation). Feel free to upload images of your own to see how well the model performs.
+ Find the notes and annotated code in the Jupyter Notebook - [`00. Unet using smp library_github`](https://github.com/Martinmbiro/unet-crack-segmentation/blob/main/00.%20Unet%20using%20smp%20library-github.ipynb). Feel free to open and explore in Google Colab

## Technical details ⚙️
Here are some technical details to note with regard to the project implementation:
#### **1. Semantic segmentation**
This is a semantic segmentation task. The goal is to segment without differentiating separate cracks on different parts of the image.
#### **2. Dataset**
The dataset used for training, validation and testing is the Crack Segmentation Dataset available on Kaggle. Find the dataset [linked here](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset). Images included in the training, validation and test sets were `9603`, `1526` and `169` respectively. Majority of these are images of walls, floors, pavements, and roads.
#### **3. Loss function**
The industry standard to U-Net image segmentation is a combined loss of _BinaryCrossEntropy Loss_ and _Dice Loss_. Hence, I combined [`binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.binary_cross_entropy_with_logits.html#torch-nn-functional-binary-cross-entropy-with-logits) from the [`torch`](https://pytorch.org/) library and [`DiceLoss`](https://smp.readthedocs.io/en/latest/losses.html#diceloss) class from [`segmentation_models_pytorch`](https://smp.readthedocs.io/en/latest/models.html)
#### **4. The model**
The U-Net model used for this task is built from the [`segmentation_models_pytorch`](https://smp.readthedocs.io/en/latest/models.html). The [ResNet-18](https://huggingface.co/papers/1512.03385) encoder backbone comes pre-trained on the [Imagenet](https://www.image-net.org/) dataset
#### **5. Training**
[Pytorch lightning](https://lightning.ai/docs/pytorch/stable/) was used for training, as opposed to traditional `for` loops. Through experimentation, the best model was achieved by:
+ Freezing the `encoder` parameters, and only training the `decoder` and `segmentation_head` layers of the U-Net.
+ An [`AdamW`](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html#adamw) optimizer with a learning rate of `1e-2`
+ A [`cosine_schedule_with_warmup`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup) scheduler from the `transformers` library with number of warming steps equal to 20% of total training steps
+ Training the model for `7` epochs.
> _At the end of the training, the model scored a training loss of `0.298904`, dice score of `0.637031` and Mean IoU of `0.580428` on the validation dataset. The trained model weights were saved to my Hugging Face Model repository [linked here](https://huggingface.co/martinmbiro/resnet18-unet-crack-segmentation)_
#### **6. Evaluation metrics**
At the end of each training epoch, the model was evaluated against the validation dataset. The [`DiceScore`](https://lightning.ai/docs/torchmetrics/stable/segmentation/dice#dice-score) and [`MeanIoU`](https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou#mean-intersection-over-union-miou) classes from [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable) were used to evaluate the model. Note that for these two classes, the following arguments were passed as follows:
+ `num_classes=2`
+ `include_background=True`
+ `input_format='index'`
#### **7. [Web App](https://huggingface.co/spaces/martinmbiro/crack-segmentation)** 💻
+ The `space` directory contains all the files required to run and deploy the Web Application on Hugging Face Spaces: `app.py` (the Gradio application and inference code), `requirements.txt` (Python dependencies), `README.md` (Space configuration and description), and `model_loader.py` (the code to download and and load the saved model weights)
+ It serves as the self-contained deployment directory that [GitHub Actions](https://docs.github.com/en/actions) syncs to Hugging Face, keeping deployment files separate from everything else.

ℹ️ **Disclaimer**  
> _The Web App uses the free **ZeroGPU** hardware which is allocated a daily 5 minutes quota. An error message will be displayed once the daily quota is exhausted_ 💔

#### 8. `.github/workflows/sync-to-hf.yaml` 📄
+ The YAML file defines the [GitHub Actions](https://docs.github.com/en/actions) workflow that automatically syncs the contents of the `space` directory to my Hugging Face [Space]() whenever changes are pushed to the main branch or the workflow is triggered manually.
+ Find Hugging Face Hub reference for [GitHub Actions](https://docs.github.com/en/actions) linked [here](https://huggingface.co/docs/hub/main/en/repositories-github-actions)

## Tools 🛠️
**The tools used for this project are:**
1. [Google Colab](https://colab.google/) - A hosted _Jupyter Notebook_ service by Google.
2. [Hugging Face](https://huggingface.co/) - An open-source platform and community hub for sharing, hosting, and building ML models, datasets, and AI applications
3. [`gradio`](https://gradio.app/) - an open-source Python library for sharing and building interactive web-based user interfaces for ML models
4. [`torch`](https://pytorch.org/) - (PyTorch) - An open-source **Deep Learning** library
5. [`segmentation_models_pytorch`](https://smp.readthedocs.io/en/latest/models.html) - A PyTorch library for easily building image segmentation models by combining decoders with pre-trained encoder backbones
6. [`lightning`](https://lightning.ai/docs/pytorch/stable/) - (PyTorch Lightning) - An open-source framework that serves as a high-level interface for PyTorch training and scaling
7. [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable) - An open-source PyTorch library that provides implementation for 100+ evaluation metrics with built-in support for automatic batch accumulation
8. [`torchinfo`](https://github.com/TylerYep/torchinfo) - A library for inspecting neural network layer structures and network parameter details
9. [`kagglehub`](https://github.com/Kaggle/kagglehub) - A library that provides a simple way to interact with [Kaggle](https://www.kaggle.com/) resources such as datasets, models, and notebook outputs in Python
10. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
11. [`PIL`](https://pillow.readthedocs.io/en/stable/) - A Python library for image manipulation and basic image processing tasks
12. [`numpy`](https://numpy.org/) - The fundamental package for scientific computing with Python
13. [`pandas`](https://pandas.pydata.org/) - An open-source tool for data analysis and manipulation in Python

> _Notable modules from [The Python Standard Library](https://docs.python.org/3/library/index.html) include [`shutil`](https://docs.python.org/3/library/shutil.html), [`pathlib`](https://docs.python.org/3/library/pathlib.html), and [`os`](https://docs.python.org/3/library/os.html)_

## Visualizations 📈
+ _Random Image / Mask pairs from training set_
> <img src='https://github.com/Martinmbiro/unet-crack-segmentation/raw/refs/heads/main/plots/training_images_masks.png' alt='random_img_mask_pairs' width=650/>

+ _Training Logs_
> <img src='https://github.com/Martinmbiro/unet-crack-segmentation/raw/refs/heads/main/plots/training_logs.png' alt='training_logs' width=450 />

+ _Training Loss [BCE Loss + DiceLoss] and Metrics_
> <img src='https://github.com/Martinmbiro/unet-crack-segmentation/raw/refs/heads/main/plots/training_metric_loss.png' alt='training_loss_and_metrics' width=650 />

+ _Test Image, Test Mask vs Predicted Mask (x, y, y-pred)_
> <img src="https://github.com/Martinmbiro/unet-crack-segmentation/raw/refs/heads/main/plots/test_image_mask_pred.png" alt="test_image_mask_predicted_mask'" width=650/>

+ _Screenshot of Web UI_
> <img src="https://github.com/Martinmbiro/unet-crack-segmentation/raw/refs/heads/main/plots/Screenshot%202026-10-03%20183222.png" alt="web-ui_screenshot" width=650/>

## Roadmap 🎌
<center>
<img src='https://github.com/Martinmbiro/variational-autoencoder/raw/refs/heads/main/plots/roadmap.png' alt='auto_encoder_sketch' width=900/>
</center>
<p align='center'><i>This project is part of a roadmap for understanding Generative Computer Vision</i></p>

## References 📚
+ [Lee, W.-M. (2025). _Hugging Face in Action._ Simon and Schuster.](https://www.manning.com/books/hugging-face-in-action)
+ _U-Net_. (n.d.). U-Net. Retrieved October 3, 2026, from https://nn.labml.ai/unet/index.html
+ [UNet: the 2015 model with 118k+ citations that changed segmentation - And how GenAI brought it back](https://www.youtube.com/watch?v=9KvngtchNww&t=4572s&pp=ygUMdW5ldCB2aXp1YXJh) - by [Vizuara AI Labs](https://vizuara.ai/)
+ [UNet and its Family: UNet++, Residual UNet, and Attention UNet | Computer Vision Bootcamp](https://www.youtube.com/watch?v=FmbxcFLSq1k&t=3849s) - by [Vizuara AI Labs](https://vizuara.ai/)
