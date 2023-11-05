# external module
# internal module
import preprocess

# Khởi tạo các thành phần tiền xử lý: model, image preprocessing
MBV3_MODEL_NAME = "mobinetv3"

mbv3_trained_model = preprocess.load_model(num_classes=2, model_name=MBV3_MODEL_NAME)

preprocess_tranform = preprocess.image_preprocess_transform()
