from PIL import Image
import numpy as np


def save_image(image_tensor, out_name, last_acti="sigmoid"):
    """
    save a single image

    :param image_tensor: torch tensor with size=(3, h, w)
    :param out_name: path+name+".jpg"
    :param last_acti: active function, tanh or sigmoid
    :return: None
    """
    if len(image_tensor.size()) == 3:
        image_numpy = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
        # image_numpy = image_tensor.cpu().detach().numpy()
        if last_acti == "tanh":
            image_numpy = (((image_numpy + 1) / 2) * 255).astype(np.uint8)
        elif last_acti == "sigmoid":
            image_numpy = (image_numpy * 255).astype(np.uint8)

        img = Image.fromarray(image_numpy)
        img.save(out_name)
    else:
        raise ValueError("input tensor not with size (3, h, w)")
    return None