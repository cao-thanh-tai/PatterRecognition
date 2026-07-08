"""Các hàm suy luận cho webcam và ảnh của bài toán nhận diện ngôn ngữ ký hiệu."""
import csv
import os
from PIL import Image

from torchvision import transforms
import torch
from ultralytics import YOLO
from src.m2 import SignLanguageModel
import cv2
import time
from collections import Counter
import numpy as np



model_1 = YOLO('models/best_model_m1_fine_tune.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_2 = SignLanguageModel(num_classes=29)
model_2.load_state_dict(torch.load('models/best_model_m2.pth', weights_only=True, map_location=device))
model_2.to(device)
model_2.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   # Khớp với kích thước khi train.
    transforms.ToTensor(),           # Chuyển ảnh PIL sang tensor đã chuẩn hóa.
    transforms.Normalize(            # Áp dụng chuẩn hóa ImageNet.
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Ánh xạ từ chỉ số sang nhãn.
idx_to_letter = {}
with open("label_mapping.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Bỏ qua dòng tiêu đề.
    for row in reader:
        idx, letter = int(row[0]), row[1]
        idx_to_letter[idx] = letter

# Cấu hình làm mượt đầu ra.
BUFFER_SIZE = 12       # Bỏ phiếu trên 12 frame gần nhất.
CONFIDENCE_THRESHOLD = 0.7  # Cần ít nhất 70% đồng thuận.
COOLDOWN_TIME = 1.0    # Chờ trước khi nhận ký tự tiếp theo.

# Trạng thái chạy.
output_buffer = []     # Bộ đệm tạm của các dự đoán.
last_detected_letter = None  # Nhãn cuối cùng đã được chốt.
last_detected_time = 0       # Mốc thời gian chốt nhãn gần nhất.
final_sentence = ""    # Câu hoàn chỉnh đang ghép dần.


def crop_hand_with_padding(img, box):
    """
    Cắt vùng bàn tay từ ảnh gốc.

    Một nền trắng kích thước 244x244 sẽ được tạo ra và vùng cắt được dán vào giữa.
    Args:
        img: Ảnh gốc từ webcam.
        box: Bounding box của YOLO (r.boxes[i]).
        padding_ratio: Tỷ lệ đệm do nơi gọi truyền vào.
        
    Returns:
        hand_crop: Ảnh bàn tay đã cắt, hoặc None nếu cắt lỗi.
    """
    # Lấy tọa độ khung.
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Tính kích thước khung.
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Tạo nền trắng 244x244.
    hand_crop = 255 * np.ones((244, 244, 3), dtype=np.uint8)
    
    # Canh vùng cắt vào giữa nền.
    center_x = 244 // 2
    center_y = 244 // 2
    
    # Tính tọa độ dán ảnh.
    paste_x1 = center_x - box_w // 2
    paste_y1 = center_y - box_h // 2
    paste_x2 = paste_x1 + box_w
    paste_y2 = paste_y1 + box_h
    
    # Dán vùng bàn tay lên nền trắng.
    hand_crop[paste_y1:paste_y2, paste_x1:paste_x2] = img[y1:y2, x1:x2]
    
    # Trả về None nếu vùng cắt rỗng.
    if hand_crop.size == 0:
        return None
    return hand_crop


def crop_hand_with_padding_v2(img, box, padding_ratio=0.15):
    """
    Cắt vùng bàn tay từ ảnh gốc với phần đệm mở rộng.

    Args:
        img: Ảnh gốc từ webcam.
        box: Bounding box của YOLO (r.boxes[i]).
        padding_ratio: Tỷ lệ đệm áp dụng quanh khung.
        
    Returns:
        hand_crop: Ảnh bàn tay đã cắt, hoặc None nếu cắt lỗi.
    """
    # Lấy tọa độ khung.
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Tính kích thước khung.
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Tính phần đệm theo tỷ lệ cấu hình.
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # Giới hạn tọa độ sau khi thêm đệm trong biên ảnh.
    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(img.shape[1], x2 + pad_w)
    new_y2 = min(img.shape[0], y2 + pad_h)
    
    # Cắt vùng bàn tay đã mở rộng.
    hand_crop = img[new_y1:new_y2, new_x1:new_x2]
    
    # Trả về None nếu vùng cắt rỗng.
    if hand_crop.size == 0:
        return None
    
    return hand_crop    

def video_stream():
    results = model_1.predict(source=0, show=False, conf=0.5, iou=0.3, stream=True)
    global last_detected_letter, last_detected_time, final_sentence, output_buffer
    
    # Xóa các ảnh kết quả cũ trước khi bắt đầu.
    output_dir = "outputs/images"
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Không thể xóa file {file_path}. Lỗi: {e}")
    for r in results:
        img = r.orig_img  # Khung hình gốc từ webcam.
        display_img = r.plot()
        
        # Xử lý từng khung bàn tay được phát hiện.
        for box in r.boxes:
            # Lấy tọa độ khung ở dạng số nguyên.
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Cắt vùng bàn tay từ ảnh gốc.
            hand_crop = crop_hand_with_padding_v2(img, box, padding_ratio=0.15)
            
            # Bỏ qua nếu vùng cắt không hợp lệ.
            if hand_crop is None:
                continue
            
            hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(hand_crop_rgb)
            # Tiền xử lý ảnh cho bộ phân loại.
            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0) # Thêm chiều batch.
            
            
            # Chạy mô hình phân loại.
            with torch.no_grad():
                input_batch = input_batch.to(device)
                output = model_2(input_batch)
                
            # Lấy lớp dự đoán.
            _, predicted_class = torch.max(output, 1)
            
            current_prediction = predicted_class.item() # Chỉ số lớp.
        
            # Thêm dự đoán hiện tại vào bộ đệm bỏ phiếu.
            output_buffer.append(current_prediction)
            
            # Giữ bộ đệm trong giới hạn cấu hình.
            if len(output_buffer) > BUFFER_SIZE:
                output_buffer.pop(0)
                
            # Kiểm tra xem nhãn đã ổn định để chốt chưa.
            if len(output_buffer) == BUFFER_SIZE:
                # Đếm tần suất nhãn trong bộ đệm.
                counter = Counter(output_buffer)
                most_common_letter, count = counter.most_common(1)[0]
                
                # Tính tỷ lệ đồng thuận.
                confidence = count / BUFFER_SIZE
                
                # Áp dụng ngưỡng chấp nhận và thời gian chờ.
                current_time = time.time()
                if confidence >= CONFIDENCE_THRESHOLD and (current_time - last_detected_time) > COOLDOWN_TIME:
                    
                    # Chỉ nhận nhãn mới để tránh lặp lại.
                    if most_common_letter != last_detected_letter:
                        last_detected_letter = most_common_letter
                        last_detected_time = current_time
                        
                        # Ánh xạ chỉ số sang nhãn.
                        letter_char = idx_to_letter[most_common_letter]
                        
                        # Xử lý các nhãn điều khiển đặc biệt.
                        if letter_char == "del":
                            final_sentence = final_sentence[:-1]  # Xóa ký tự cuối.
                        elif letter_char == "space":
                            final_sentence += " "  # Thêm dấu cách.
                        elif letter_char == "nothing":
                            pass
                        else:
                        # Ghép ký tự vừa nhận vào câu.
                            final_sentence += letter_char
                        # Lưu ảnh đã chốt để đối chiếu sau.
                        cv2.imwrite(f"outputs/images/{letter_char}_{int(time.time())}.jpg", hand_crop)
                        
                        print("\n" + "="*40)
                        print(f"Đã chốt chữ: {letter_char} (độ tự tin: {confidence*100:.1f}%)")
                        print(f"Câu hiện tại: {final_sentence}")
                        print("="*40 + "\n")
                
        # Vẽ nền tối cho phần chữ hiển thị.
        cv2.rectangle(display_img, (10, 10), (600, 60), (0, 0, 0), -1)
        
        # Hiển thị câu hiện tại trên khung hình.
        cv2.putText(display_img, f"TEXT: {final_sentence}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Webcam - Nhấn 'q' để thoát", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đã đóng webcam theo yêu cầu người dùng.")
            break # Thoát khỏi vòng lặp khung hình.
    del results
    cv2.destroyAllWindows()
    print("Kết thúc luồng video. Câu hoàn chỉnh:", final_sentence)
    
    return final_sentence  # Trả về câu hoàn chỉnh sau khi kết thúc luồng video.

def image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = preprocess(img_pil)
    input_batch = input_tensor.unsqueeze(0) # Thêm chiều batch.
    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model_2(input_batch)
    _, predicted_class = torch.max(output, 1)
    print(f"Chỉ số nhãn dự đoán: {predicted_class.item()}")
    return idx_to_letter[predicted_class.item()]  # Trả về ký tự dự đoán từ ảnh.

