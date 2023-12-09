import process
import cv2

# test for a single file
def processAnswerSheet(imgPath):
    # Lấy ảnh 
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)[:, :, ::-1]
    
    # Cắt background
    document = process.extract(image_true=image)
    document = document / 255.0
    
    # Resize lại ảnh
    resize_img = cv2.resize(document, (1056, 1500), interpolation=cv2.INTER_AREA)
    
    # cv2.imwrite("test_extract.jpg", document*255)
    
    # cv2.imshow("Image", resize_img)
    # cv2.waitKey(0)
    
    # Lấy thông tin mã người làm bài và mã đề thi
    result_info = process.get_info(resize_img)
    
    number_answer = 120
    
    # Lấy các câu trả lời 
    # result_answer = get_answer(resize_img, number_answer)


    print(result_info)
    
    
    return result_info



if __name__ == "__main__":
    
    # test 
    result_info = processAnswerSheet('./data/f13.jpg')
    
    # print("Main Application")
    # data_path = "./data";
    # numFiles = 0;
    # for file in os.scandir(data_path):
    #     if file.is_file():
    #         numFiles += 1
            
            
    
    # process each image file in folder data
    # for i in range (numFiles):
    #     image_test_path = f"./data/f{i+1}.jpg"
    #     image_test = cv2.imread(image_test_path, cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
    #     # extract background
    #     document = process.extract(image_test)
    #     document = document / 255.0;
    #     form_write_path = f"./processed_data/cropped_form/form_cropped_{i+1}.jpg"
    #     cv2.imwrite(form_write_path, document * 255.0)
        
    #     # resize image
    #     resize_image = cv2.resize(document, (1056, 1549), interpolation=cv2.INTER_AREA)
        
    #     # Crop info section
    #     # cropped_section = process.crop_info_section(resize_image)
    #     # info_write_path = f"./processed_data/cropped_info_section/info_section_cropped_{i+1}.jpg"
    #     # cv2.imwrite(info_write_path, cropped_section * 255.0)
        
    #     # Crop multiple choice section
    #     process.get_answers_boxes(resize_image, i+1)
