import os
import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
from PIL import Image

# Загружаем предобученную модель R-CNN
model = fasterrcnn_resnet50_fpn()
model.eval()

# Преобразование изображений для модели
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Функция для Selective Search
def selective_search(image_path):
    image = cv2.imread(image_path)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return image, rects

# Папка для результатов
result_folder = "result"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Путь к папке с изображениями
image_folder = "recog_2"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Основная логика
for image_path in image_paths:
    image, proposals = selective_search(image_path)
    print(f"Обработано изображение: {image_path}")
    print(f"Количество гипотез: {len(proposals)}")

    filtered_proposals = []
    threshold = 0.6  # Порог уверенности

    # Обработка каждого прямоугольника
    for (x, y, w, h) in proposals:  # Ограничиваем до 2000 гипотез
        proposal_img = image[y:y+h, x:x+w]
        proposal_tensor = transform(proposal_img).unsqueeze(0)  # Добавляем размерность для батча

        with torch.no_grad():
            predictions = model(proposal_tensor)

        # Проверяем уверенность
        scores = predictions[0]['scores']
        boxes = predictions[0]['boxes']

        for score, box in zip(scores, boxes):
            if score > threshold:
                filtered_proposals.append((box, score.item()))

    # Рисуем рамки для всех гипотез
    for (box, conf) in filtered_proposals:
        x1, y1, x2, y2 = box.int().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Сохраняем результатное изображение
    result_image_path = os.path.join(result_folder, os.path.basename(image_path))
    cv2.imwrite(result_image_path, image)
    print(f"Сохранено результатное изображение: {result_image_path}")
