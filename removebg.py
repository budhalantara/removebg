import torch
from u2net import U2NET
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Tuple
import numpy as np
import os


def normPRED(d):
    # normalize the predicted SOD probability map
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def normalize_image(
    img: Image,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    img = img.convert("RGB").resize(size, Image.LANCZOS)

    img_arr = np.array(img)
    img_arr = img_arr / np.max(img_arr)

    tmp_img = np.zeros((img_arr.shape[0], img_arr.shape[1], 3))
    tmp_img[:, :, 0] = (img_arr[:, :, 0] - mean[0]) / std[0]
    tmp_img[:, :, 1] = (img_arr[:, :, 1] - mean[1]) / std[1]
    tmp_img[:, :, 2] = (img_arr[:, :, 2] - mean[2]) / std[2]

    tmp_img = tmp_img.transpose((2, 0, 1))
    return np.expand_dims(tmp_img, 0).astype(np.float32)


def load_model():
    net = U2NET(3, 1)
    model_path = 'u2net.pth'

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.eval()
    return net


def remove_bg(image: Image):
    normalized = normalize_image(img=image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=(320, 320))

    input = torch.from_numpy(normalized)

    model = load_model()
    d1, d2, d3, d4, d5, d6, d7 = model(input)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    mask_image = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize(image.size, Image.LANCZOS)

    result_image = Image.new("RGBA", (image.size), 0)
    result_image = Image.composite(image.convert("RGBA"), result_image, mask_image)
    return result_image


def process(image_base_dir: str, result_base_dir: str, category: str):
    image_path = os.path.join(image_base_dir, top)
    image = Image.open(image_path)

    result = remove_bg(image)

    os.makedirs(os.path.join(result_base_dir, category), exist_ok=True)

    filename = os.path.splitext(top)[0] + ".png"
    result_path = os.path.join(result_base_dir, category, filename)
    result.save(result_path)


result_base_dir = "./result"

tops_base_dir = "./images/tops"
for top in os.listdir(tops_base_dir):
    process(tops_base_dir, result_base_dir, "tops")

shoes_base_dir = "./images/shoes"
for top in os.listdir(shoes_base_dir):
    process(shoes_base_dir, result_base_dir, "shoes")

bottoms_base_dir = "./images/bottoms"
for top in os.listdir(bottoms_base_dir):
    process(bottoms_base_dir, result_base_dir, "bottoms")
