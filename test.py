# test yolo trc web camp
import csv
from PIL import Image

from torchvision import transforms
import torch
from ultralytics import YOLO
from src.m2 import SignLanguageModel
import cv2
import time
from collections import Counter



model_1 = YOLO('models/best_model_m1_fine_tune.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_2 = SignLanguageModel(num_classes=29)
model_2.load_state_dict(torch.load('models/best_model_m2.pth'))
model_2.to(device)
model_2.eval()

preprocess = transforms.Compose([
    # transforms.ToPILImage(),         # Chuyển từ mảng Numpy (OpenCV) sang ảnh PIL để transforms hiểu
    transforms.Resize((224, 224)),   # Đổi kích thước về chuẩn lúc train (ví dụ ResNet thường là 224x224)
    transforms.ToTensor(),           # CỰC KỲ QUAN TRỌNG: Chuyển ảnh từ PIL sang PyTorch TENSOR và scale về [0, 1]
    transforms.Normalize(            # Chuẩn hóa chuẩn ImageNet (hoặc theo tập train của nhóm bạn)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# lấy mapping từ index sang chữ cái
idx_to_letter = {}
with open("label_mapping.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Bỏ qua tiêu đề
    for row in reader:
        idx, letter = int(row[0]), row[1]
        idx_to_letter[idx] = letter

# --- CẤU HÌNH THUẬT TOÁN LỌC OUTPUT ---
BUFFER_SIZE = 12       # Lưu kết quả của 12 frame gần nhất để bầu chọn
CONFIDENCE_THRESHOLD = 0.7  # Tỉ lệ đồng thuận phải trên 70% thì mới chốt chữ
COOLDOWN_TIME = 1.0    # Cần ít nhất 1 giây để ghi nhận một chữ tiếp theo (tránh double click chữ)

# Các biến trạng thái chạy ngầm
output_buffer = []     # Bộ nhớ tạm lưu chữ
last_detected_letter = None  # Chữ cái cuối cùng được chốt thành công
last_detected_time = 0       # Mốc thời gian chốt chữ gần nhất
final_sentence = ""    # Câu hoàn chỉnh sau khi ghép các chữ cái lại


def crop_hand_with_padding(img, box, padding_ratio=0.15):
    """
    Hàm cắt vùng bàn tay từ ảnh gốc và nới rộng ra theo tỉ lệ mong muốn.
    
    Args:
        img: Ảnh gốc (mảng numpy từ webcam)
        box: Bounding box của YOLO (r.boxes[i])
        padding_ratio: Tỉ lệ nới rộng (0.15 tương đương 15%)
        
    Returns:
        hand_crop: Vùng ảnh bàn tay đã nới rộng, hoặc None nếu cắt lỗi
    """
    # Lấy tọa độ gốc từ YOLO
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # 1. Tính chiều rộng và chiều cao của khung bàn tay hiện tại
    box_w = x2 - x1
    box_h = y2 - y1
    
    # 2. Tính độ rộng cần nới ra 4 hướng
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # 3. Tiến hành nới rộng tọa độ
    x1_pad = x1 - pad_w
    y1_pad = y1 - pad_h
    x2_pad = x2 + pad_w
    y2_pad = y2 + pad_h
    
    # 4. CHẶN LỖI TRÀN VIỀN: Ép tọa độ luôn nằm trong giới hạn của bức ảnh gốc
    img_h, img_w, _ = img.shape
    x1_pad = max(0, x1_pad)
    y1_pad = max(0, y1_pad)
    x2_pad = min(img_w, x2_pad)
    y2_pad = min(img_h, y2_pad)
    
    # 5. Cắt ảnh
    hand_crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Nếu ảnh cắt ra bị rỗng thì trả về None
    if hand_crop.size == 0:
        return None
    # debug_view = cv2.resize(hand_crop, (300, 300))
    # cv2.imshow("Anh thuc te ném vao ResNet", debug_view)
    return hand_crop

def video_stream():
    results = model_1.predict(source=0, show=False, conf=0.5, iou=0.3, stream=True)
    global last_detected_letter, last_detected_time, final_sentence, output_buffer
    for r in results:
        img = r.orig_img  # Đây là ẢNH GỐC to đùng từ webcam
        display_img = r.plot()
        
        # Duyệt qua từng khung hình vuông mà YOLO tìm thấy bàn tay
        for box in r.boxes:
            # Lấy tọa độ (x_min, y_min, x_max, y_max) dạng số nguyên
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # CÂY ĐŨA THẦN Ở ĐÂY: Cắt riêng vùng bàn tay ra khỏi ảnh gốc
            # Trong OpenCV/Numpy, cắt ảnh theo thứ tự: ảnh[y_min:y_max, x_min:x_max]
            hand_crop = crop_hand_with_padding(img, box, padding_ratio=0.15)
            
            # Nếu cắt bị lỗi hoặc ảnh trống thì bỏ qua
            if hand_crop is None:
                continue
            
            hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(hand_crop_rgb)
            # Tiến hành xử lý input cho hand_crop
            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0) # Thêm chiều batch size -> [1, 3, 224, 224]
            
            
            # Ném vùng tay đã cắt vào model_2
            with torch.no_grad():
                input_batch = input_batch.to(device)
                output = model_2(input_batch)
                
            # Lấy kết quả dự đoán ký hiệu
            _, predicted_class = torch.max(output, 1)
            
            current_prediction = predicted_class.item() # Giả sử trả về số hoặc chữ: ví dụ 'A'
        
            # 1. Thêm kết quả của frame hiện tại vào bộ đệm
            output_buffer.append(current_prediction)
            
            # Nếu bộ đệm vượt quá kích thước cấu hình thì xóa bớt thằng cũ nhất
            if len(output_buffer) > BUFFER_SIZE:
                output_buffer.pop(0)
                
            # 2. KIỂM TRA ĐIỀU KIỆN ĐỂ CHỐT CHỮ
            if len(output_buffer) == BUFFER_SIZE:
                # Đếm xem mỗi chữ xuất hiện bao nhiêu lần trong bộ đệm
                counter = Counter(output_buffer)
                most_common_letter, count = counter.most_common(1)[0]
                
                # Tính tỉ lệ đồng thuận (ví dụ: xuất hiện 9/12 frame -> ~75%)
                confidence = count / BUFFER_SIZE
                
                # Nếu đạt đủ độ tin tưởng và đã qua thời gian Cooldown
                current_time = time.time()
                if confidence >= CONFIDENCE_THRESHOLD and (current_time - last_detected_time) > COOLDOWN_TIME:
                    
                    # CHỈ LẤY CHỮ MỚI (Tránh việc giữ nguyên tay nó cứ cộng dồn chữ AAAAAAA liên tục)
                    if most_common_letter != last_detected_letter:
                        last_detected_letter = most_common_letter
                        last_detected_time = current_time
                        
                        # mapping từ index sang chữ cái
                        letter_char = idx_to_letter[most_common_letter]
                        # Cộng chữ mới vào câu hoàn chỉnh
                        final_sentence += letter_char
                        # lưu ảnh chốt vào thư mục output/images
                        cv2.imwrite(f"outputs/images/{letter_char}_{int(time.time())}.jpg", hand_crop)
                        
                        print("\n" + "="*40)
                        print(f"🎯 ĐÃ CHỐT CHỮ: {letter_char} (Độ tự tin: {confidence*100:.1f}%)")
                        print(f"📝 CÂU HIỆN TẠI: {final_sentence}")
                        print("="*40 + "\n")
                # print(f"Ký hiệu nhận diện được: {predicted_class.item()}")
                
        # Vẽ một cái nền đen mờ ở góc trên để chữ nổi bật, dễ nhìn hơn
        cv2.rectangle(display_img, (10, 10), (600, 60), (0, 0, 0), -1)
        
        # Ghi câu hiện tại lên góc trái màn hình webcam
        # text: "TEXT: <câu của ông>"
        cv2.putText(display_img, f"TEXT: {final_sentence}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Webcam - Nhấn 'q' để thoát", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tắt cụp webcam!")
            break # Bẻ gãy vòng lặp for ngay lập tức
    del results
    cv2.destroyAllWindows()
    print("Kết thúc luồng video, trả về câu hoàn chỉnh:", final_sentence)

def image(img):
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # Thêm chiều batch size -> [1, 3, 224, 224]
    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model_2(input_batch)
    _, predicted_class = torch.max(output, 1)
    print(f"Ký hiệu nhận diện được: {predicted_class.item()}")

