import process
import cv2
import os

if __name__ == "__main__":
    
    # test 
    img_path = "C:\\Users\\leope\\Desktop\\answer_new_data\\sheet26.jpg";
    local_img_path = "./data/f13.jpg"
    result_info = process.process_answer_sheet(imgPath = local_img_path)
    
    # print("Main Application")
    # data_path = "C:\\Users\\leope\\Desktop\\answer_new_data";
    # numFiles = 0;
    # for file in os.scandir(data_path):
    #     if file.is_file():
    #         numFiles += 1
            
            
    
    # process each image file in folder data
    # for i in range (numFiles):
        # image_test_path = f"./data/f{i+1}.jpg"
        # image_test = cv2.imread(image_test_path, cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
        # # extract background
        # document = process.extract(image_test)
        # document = document / 255.0;
        # form_write_path = f"./processed_data/cropped_form/form_cropped_{i+1}.jpg"
        # cv2.imwrite(form_write_path, document * 255.0)
        
        # # resize image
        # resize_image = cv2.resize(document, (1056, 1549), interpolation=cv2.INTER_AREA)
        
        # # Crop info section
        # # cropped_section = process.crop_info_section(resize_image)
        # # info_write_path = f"./processed_data/cropped_info_section/info_section_cropped_{i+1}.jpg"
        # # cv2.imwrite(info_write_path, cropped_section * 255.0)
        
        # # Crop multiple choice section
        # process.get_answers_boxes(resize_image, i+1)
        
        # Generate data
        # if i < 24: continue
        # # image_test_path = f"./data/f{i+1}.jpg"
        # image_test_path = f"C:\\Users\\leope\\Desktop\\answer_new_data\\sheet{i+1}.jpg"
        # image_test = cv2.imread(image_test_path, cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
        # # extract background
        # document = process.extract(image_test)
        # document = document / 255.0;
        # # form_write_path = f"./processed_data/cropped_form/form_cropped_{i+1}.jpg"
        # # cv2.imwrite(form_write_path, document * 255.0)
        
        # # resize image
        # resize_image = cv2.resize(document, (1056, 1549), interpolation=cv2.INTER_AREA)
        
        # # cropped_section = process.crop_info_section(resize_image)
        # # info_write_path = f"./processed_data/cropped_info_section/info_section_cropped_{i+1}.jpg"
        # # cv2.imwrite(info_write_path, cropped_section * 255.0)
        # # box_info = process.get_info(resize_image)
        
        # # Crop multiple choice section
        # process.get_answers_boxes(resize_image, i+1)
        
        
