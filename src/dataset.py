# dataloader for sign language recognition
# file có dạng tên file là chữ cái tương ứng với ngôn ngữ ký hiệu đó và bên trong là các bức ảnh tương ứng với chữ cái đó và đc lưu vd A1.jpg, B6.jpg
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the sign language letters (folders).
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Create label mapping
        labels_set = sorted([label for label in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, label))])
        for idx, label in enumerate(labels_set):
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label
        
        # Load image paths and labels
        for label in labels_set:
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(label_dir, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(self.label_to_idx[label])
    
    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Return image and label at given index.
        
        Args:
            idx (int): Index of the image.
            
        Returns:
            tuple: (image, label) where label is integer index.
        """
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {str(e)}")
    
    def get_label_name(self, label_idx):
        """Convert label index back to label name."""
        return self.idx_to_label.get(label_idx, None)
    
    def get_num_classes(self):
        """Return number of classes."""
        return len(self.label_to_idx)
    
    
# dataloader for hand detection
# gồm 1 file ảnh, và 1 file label tương ứng với ảnh đó, file label có dạng txt và bên trong là các tọa độ của bounding box của bàn tay trong ảnh đó
# trong file label có 2 file là YOLO và VOC, mình sẽ dùng YOLO
# file txt có dạng: class_id x_center y_center width height (all normalized to [0, 1]), và thể có nhiều bounding box trong 1 file txt, mỗi bounding box là 1 dòng
# cấu trúc file: root_dir/
# ├── images
# └── labels/
#     └── image1.txt
class HandDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        
        # Load image paths and corresponding label paths
        for file_name in os.listdir(os.path.join(root_dir, 'images')):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root_dir, 'images', file_name)
                label_name = os.path.splitext(file_name)[0] + '.txt'
                label_path = os.path.join(root_dir, 'labels', label_name)
                if os.path.exists(label_path):
                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)
    
    def __len__(self):
        """Return total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Return image and bounding boxes at given index.
        
        Args:
            idx (int): Index of the image.
            
        Returns:
            tuple: (image, bounding_boxes) where bounding_boxes is a list of (class_id, x_center, y_center, width, height) tuples.
        """
        try:
            image_path = self.image_paths[idx]
            label_path = self.label_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Load bounding boxes from label file
            bounding_boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bounding_boxes.append((class_id, x_center, y_center, width, height))
            
            return image, bounding_boxes
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path} or label {label_path}: {str(e)}")