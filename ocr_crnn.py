import cv2
import numpy as np
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, numClasses):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, (3, 3), 1, (0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Linear(512, numClasses)

    def forward(self, images):
        features = self.cnn(images)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        sequence, _ = self.rnn(features)
        return self.classifier(sequence)


class PlateOCR:
    def __init__(self, modelPath, device):
        self.device = torch.device(device)
        self.checkpoint = torch.load(modelPath, map_location="cpu", weights_only=True)
        self.characters = self.checkpoint["characters"]
        self.imageWidth = self.checkpoint["image_width"]
        self.imageHeight = self.checkpoint["image_height"]
        self.indexToChar = {index + 1: character for index, character in enumerate(self.characters)}
        self.model = CRNN(len(self.characters) + 1)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def resizeTheoChieuCao(self, image, targetHeight):
        height, width = image.shape[:2]
        scale = targetHeight / max(height, 1)
        targetWidth = max(1, int(width * scale))
        return cv2.resize(image, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)

    def gheBienHaiDong(self, image):
        """Ép ảnh thành dạng biển 2 dòng (ghép nửa trên + nửa dưới thành 1 hàng)."""
        height, width = image.shape[:2]
        middle = height // 2
        top = image[:middle, :]
        bottom = image[middle:, :]

        if top.size == 0 or bottom.size == 0:
            return image

        top = self.resizeTheoChieuCao(top, 24)
        bottom = self.resizeTheoChieuCao(bottom, 24)
        gap = np.full((24, 5), 255, dtype=np.uint8)

        return np.concatenate([top, gap, bottom], axis=1)

    def chuyenBienHaiDong(self, image, forceTwoLine=None):
        height, width = image.shape[:2]

        # Vùng tỉ lệ mập mờ (biển 1 dòng dài vs 2 dòng vuông đều có thể rơi vào đây
        # tuỳ box YOLO cắt lỏng/chặt) -> để read() tự thử cả 2 cách và lấy cái tin cậy hơn,
        # thay vì đoán chắc 1 lần rồi sai là sai luôn.
        if forceTwoLine is True:
            return self.gheBienHaiDong(image)
        if forceTwoLine is False:
            return image

        if width / max(height, 1) >= 2.0:
            return image

        return self.gheBienHaiDong(image)

    def chuanHoaAnh(self, image, forceTwoLine=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.chuyenBienHaiDong(gray, forceTwoLine)

        height, width = gray.shape[:2]
        scale = min(self.imageWidth / max(width, 1), self.imageHeight / max(height, 1))

        targetWidth = max(1, int(width * scale))
        targetHeight = max(1, int(height * scale))

        gray = cv2.resize(gray, (targetWidth, targetHeight), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((self.imageHeight, self.imageWidth), 255, dtype=np.uint8)

        x = (self.imageWidth - targetWidth) // 2
        y = (self.imageHeight - targetHeight) // 2

        canvas[y:y + targetHeight, x:x + targetWidth] = gray

        tensor = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5

        return tensor.to(self.device)

    def decode(self, output):
        probabilities = torch.softmax(output[0], dim=1)
        prediction = probabilities.argmax(dim=1)

        characters = []
        confidences = []
        previousIndex = -1

        for position, index in enumerate(prediction.tolist()):
            if index != 0 and index != previousIndex:
                characters.append(self.indexToChar.get(index, ""))
                confidences.append(float(probabilities[position, index].item()))

            previousIndex = index

        text = "".join(characters)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, confidence

    def docMotLan(self, image, forceTwoLine=None):
        tensor = self.chuanHoaAnh(image, forceTwoLine)
        with torch.inference_mode():
            output = self.model(tensor)
        return self.decode(output)

    def read(self, image):
        if image is None or image.size == 0:
            return None, 0.0

        height, width = image.shape[:2]
        ratio = width / max(height, 1)

        # Tỉ lệ rõ ràng (biển dài 1 dòng, hoặc biển gần vuông 2 dòng) -> đọc thẳng 1 lần.
        if ratio >= 2.6 or ratio <= 1.3:
            return self.docMotLan(image)

        # Tỉ lệ mập mờ (do YOLO crop lỏng/chặt) -> thử cả 2 cách, lấy kết quả tin cậy hơn
        # thay vì đoán 1 lần bằng ngưỡng cứng rồi đọc sai cả biển.
        textMotDong, confMotDong = self.docMotLan(image, forceTwoLine=False)
        textHaiDong, confHaiDong = self.docMotLan(image, forceTwoLine=True)

        if confHaiDong >= confMotDong:
            return textHaiDong, confHaiDong
        return textMotDong, confMotDong