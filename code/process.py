# external module
import gc
import cv2
import torch
import numpy as np
import imutils

# internal module
import preprocess

# Khởi tạo các thành phần tiền xử lý: model, image preprocessing
MBV3_MODEL_NAME = "mobinetv3"

mbv3_trained_model = preprocess.load_model(num_classes=2, model_name=MBV3_MODEL_NAME)

preprocess_tranform = preprocess.image_preprocess_transform()


# Sắp xếp các điểm ở góc của box
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


# Tìm các điểm góc của box
def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)


# Cắt phiếu trả lời ra khỏi ảnh input
def extract(image_true=None, image_size=640, BUFFER=10):
    global preprocess_tranform

    # image_size và image half size
    IMAGE_SIZE = image_size
    half_size = IMAGE_SIZE // 2

    # Lấy image shape
    imageH, imageW, C = image_true.shape

    # Resize lại ảnh
    image_model = cv2.resize(
        image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
    )

    scale_x = imageW / IMAGE_SIZE
    scale_y = imageH / IMAGE_SIZE

    # Tạo image_model để thực hiện train
    image_model = preprocess_tranform(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)
    with torch.no_grad():
        out = mbv3_trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    #
    out = (
        torch.argmax(out, dim=1, keepdims=True)
        .permute(0, 2, 3, 1)[0]
        .numpy()
        .squeeze()
        .astype(np.int32)
    )
    r_W, r_H = out.shape

    out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    out_extended[
        half_size : half_size + IMAGE_SIZE, half_size : half_size + IMAGE_SIZE
    ] = (out * 255)
    out = out_extended.copy()

    del out_extended
    gc.collect()

    # Sử dụng thư viện cv2 để lấy vùng chứa phiếu đã xác định được từ ảnh gốc sử dụng
    # Sử dụng Canny và Contours
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # Xử lý và lấy được toạ độ các góc
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    # Rescale ảnh về kích thước ban đầu
    corners[:, 0] -= half_size
    corners[:, 1] -= half_size

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # Test for debug
    # print(corners)
    # print(corners.min(axis=0))
    # print(corners.max(axis=0))
    # print((imageW, imageH))
    # print(corners.min(axis=0) >= (0, 0))
    # print(corners.max(axis=0) <= (imageW, imageH))
    # print(
    #     np.all(corners.min(axis=0) >= (0, 0))
    #     and np.all(corners.max(axis=0) <= (imageW, imageH))
    # )

    if not (
        np.all(corners.min(axis=0) >= (0, 0))
        and np.all(corners.max(axis=0) <= (imageW, imageH))
    ):
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        #
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imageW:
            right_pad = (box_x_max - imageW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imageH:
            bottom_pad = (box_y_max - imageH) + BUFFER

        image_extended = np.zeros(
            (top_pad + bottom_pad + imageH, left_pad + right_pad + imageW, C),
            dtype=image_true.dtype,
        )
        image_extended[
            top_pad : top_pad + imageH, left_pad : left_pad + imageW, :
        ] = image_true
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    # Xử lý toạ độ các góc -> lấy được toạ độ góc gần nhất của phiếu trong ảnh ban đầu (image_true)
    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = corners
    destination_corners = find_dest(corners)
    # print(f"Destination corner{destination_corners}")

    # Tranform và tách phiếu từ ảnh gốc
    M = cv2.getPerspectiveTransform(
        np.float32(corners), np.float32(destination_corners)
    )

    final = cv2.warpPerspective(
        image_true,
        M,
        (destination_corners[2][0], destination_corners[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )
    # Xử lý màu và round về khoảng [0:255] của ảnh đã tách
    final = np.clip(final, a_min=0.0, a_max=255.0)
    return final

 # ======= GET x,y functions ==========
def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

# =================== PROCESS INFO SECTION ===================================== 


def crop_info_section(image):
    left = 700
    top = 0
    right = 1056
    bottom = 549
    cropped_info_section = image[top:bottom, left:right]
    return cropped_info_section