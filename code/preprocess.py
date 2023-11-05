import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

# Khởi tạo device
process_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm load mô hình mobinetV3
CHECKPOINT_MBV3_MODEL_PATH = r"model/model_mbv3_iou_mix_2C049.pth"


def load_model(num_classes=1, model_name="mobinetv3"):
    # model return
    model = None
    if model_name == "mobinetv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
    else:
        model = deeplabv3_resnet50(num_classes=num_classes)

    # set model vào device và load vào device
    model.to(device=process_device)
    checkpoints = torch.load(f=CHECKPOINT_MBV3_MODEL_PATH, map_location=process_device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    loaded = model(torch.randn((2, 3, 384, 384)))
    return model


# Sử dụng torch vision để truyển đổi ảnh đầu vào theo bộ tham số
def image_preprocess_transform(
    mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)
):
    common_tranformation = torchvision_T.Compose(
        [torchvision_T.ToTensor(), torchvision_T.Normalize(mean=mean, std=std)]
    )
    return common_tranformation
