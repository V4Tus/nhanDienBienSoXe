import cv2
import os
import re
import sqlite3
import time
import torch
from collections import Counter, defaultdict, deque
from datetime import datetime
from ultralytics import YOLO
from ocr_crnn import PlateOCR

DETECTOR_MODEL_PATH = "biensoxe.pt"
OCR_MODEL_PATH = "plate_crnn_deploy.pth"
VIDEO_PATH = "test.mov"
DATABASE_PATH = "bienso.db"
OUTPUT_VIDEO_PATH = "video_ket_qua.mp4"
OUTPUT_CROP_DIR = "TEST_CROP"
DETECTOR_IMAGE_SIZE = 960
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.5

OCR_FRAME_INTERVAL = 3
OCR_HISTORY_SIZE = 7
MIN_OCR_RESULTS = 3
MIN_OCR_CONFIDENCE = 0.30

MIN_PLATE_WIDTH = 35
MIN_PLATE_HEIGHT = 12
MIN_SHARPNESS = 15.0

CROP_PADDING_X = 0.06
CROP_PADDING_Y = 0.12

os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print("Thiết bị:", DEVICE)

detector = YOLO(DETECTOR_MODEL_PATH)
ocr_reader = PlateOCR(OCR_MODEL_PATH, DEVICE)
ocr_history = defaultdict(lambda: deque(maxlen=OCR_HISTORY_SIZE))

track_results = {}
best_crops = {}
last_seen_frame = {}
saved_tracks = set()
LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8"
}
DIGIT_TO_LETTER = {
    "0": "D",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B"
}

def luuThongTinBienSo(plate, image_path):
    detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO bienso (plate, image_path, time) VALUES (?, ?, ?)", (plate, image_path, detected_time))
        connection.commit()

def tinhDoNet(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def chuyenThanhSo(character):
    if character.isdigit():
        return character, 0
    if character in LETTER_TO_DIGIT:
        return LETTER_TO_DIGIT[character], 1
    return character, 10

def chuyenThanhChu(character):
    if character.isalpha():
        return character, 0
    if character in DIGIT_TO_LETTER:
        return DIGIT_TO_LETTER[character], 1
    return character, 10

def taoUngVienBienSo(text):
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(clean) < 7 or len(clean) > 9:
        return None
    candidates = []
    for letter_count in (1, 2):
        digit_count = len(clean) - 2 - letter_count
        if digit_count not in (4, 5):
            continue
        result = []
        correction_cost = 0
        valid = True
        for index, character in enumerate(clean):
            if index < 2:
                converted, cost = chuyenThanhSo(character)
            elif index < 2 + letter_count:
                converted, cost = chuyenThanhChu(character)
            else:
                converted, cost = chuyenThanhSo(character)
            if cost >= 10:
                valid = False
                break
            result.append(converted)
            correction_cost += cost
        if not valid:
            continue
        plate = "".join(result)
        pattern = (rf"^[0-9]{{2}}" rf"[A-Z]{{{letter_count}}}" rf"[0-9]{{{digit_count}}}$")
        if re.fullmatch(pattern, plate):
            candidates.append((plate, correction_cost))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1])
    return candidates[0][0]

def dinhDangBienSo(plate):
    if not plate:
        return ""
    if len(plate) == 7:
        return (f"{plate[:3]}-" f"{plate[3:]}")
    if len(plate) == 8:
        return (f"{plate[:3]}-" f"{plate[3:6]}." f"{plate[6:]}")
    if len(plate) == 9:
        return (f"{plate[:4]}-" f"{plate[4:7]}." f"{plate[7:]}")
    return plate

def docBienSo(crop):
    text, confidence = ocr_reader.read(crop)
    if not text:
        return None, 0.0, ""
    raw_text = re.sub(r"[^A-Z0-9]", "", text.upper())
    plate = taoUngVienBienSo(raw_text)
    return (plate, float(confidence), raw_text)

def tongHopOCR(history):
    valid_results = [result for result in history if (result["plate"] is not None and result["confidence"] >= MIN_OCR_CONFIDENCE)]
    if len(valid_results) < MIN_OCR_RESULTS:
        return None, 0.0
    length_counter = Counter(len(result["plate"]) for result in valid_results)
    best_length = length_counter.most_common(1)[0][0]
    same_length_results = [result for result in valid_results if len(result["plate"]) == best_length]
    if len(same_length_results) < MIN_OCR_RESULTS:
        return None, 0.0
    final_characters = []
    position_scores = []
    for position in range(best_length):
        character_scores = defaultdict(float)
        for result in same_length_results:
            character = result["plate"][position]
            character_scores[character] += result["confidence"]
        best_character = max(character_scores, key=character_scores.get)
        total_score = sum(character_scores.values())
        position_score = character_scores[best_character] / total_score if total_score > 0 else 0.0
        final_characters.append(best_character)
        position_scores.append(position_score)
    final_plate = "".join(final_characters)
    final_plate = taoUngVienBienSo(final_plate)
    if final_plate is None:
        return None, 0.0
    confidence = sum(position_scores) / len(position_scores)
    return final_plate, confidence

def layCropBienSo(frame, x1, y1, x2, y2):
    frame_height, frame_width = frame.shape[:2]
    plate_width = x2 - x1
    plate_height = y2 - y1
    padding_x = int(plate_width * CROP_PADDING_X)
    padding_y = int(plate_height * CROP_PADDING_Y)
    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(frame_width, x2 + padding_x)
    crop_y2 = min(frame_height, y2 + padding_y)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    raise FileNotFoundError(f"Không mở được video: {VIDEO_PATH}")
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = video.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 30.0
writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height))
frame_count = 0
previous_time = time.perf_counter()
display_fps = 0.0
try:
    while True:
        success, frame = video.read()
        if not success:
            break
        frame_count += 1
        output_frame = frame.copy()
        results = detector.track(source=frame, persist=True, tracker="bytetrack.yaml", imgsz=DETECTOR_IMAGE_SIZE, conf=DETECTOR_CONFIDENCE, iou=DETECTOR_IOU, device=DEVICE, verbose=False)
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                coordinates = boxes.xyxy.int().cpu().numpy()
                detector_confidences = boxes.conf.cpu().numpy()
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().numpy()
                else:
                    track_ids = [-1 for _ in range(len(coordinates))]
                for box, detector_confidence, track_id in zip(coordinates, detector_confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, min(x1, video_width - 1))
                    y1 = max(0, min(y1, video_height - 1))
                    x2 = max(0, min(x2, video_width))
                    y2 = max(0, min(y2, video_height))
                    plate_width = x2 - x1
                    plate_height = y2 - y1
                    if (plate_width < MIN_PLATE_WIDTH or plate_height < MIN_PLATE_HEIGHT):
                        continue
                    crop = layCropBienSo(frame, x1, y1, x2, y2)
                    if crop.size == 0:
                        continue
                    sharpness = tinhDoNet(crop)
                    last_seen_frame[track_id] = frame_count
                    current_best = best_crops.get(track_id)
                    if (current_best is None or sharpness > current_best["sharpness"]):
                        best_crops[track_id] = {
                            "image": crop.copy(),
                            "sharpness": sharpness
                        }
                    if (frame_count % OCR_FRAME_INTERVAL == 0 and sharpness >= MIN_SHARPNESS):
                        plate, ocr_confidence, raw_text = docBienSo(crop)
                        ocr_history[track_id].append({
                            "plate": plate,
                            "confidence": ocr_confidence,
                            "raw_text": raw_text,
                            "sharpness": sharpness
                        })
                        final_plate, vote_confidence = tongHopOCR(ocr_history[track_id])
                        if final_plate is not None:
                            track_results[track_id] = {
                                "plate": final_plate,
                                "confidence": vote_confidence
                            }
                            if track_id not in saved_tracks:
                                best_crop = best_crops[track_id]["image"]
                                crop_path = os.path.join(OUTPUT_CROP_DIR, (f"{final_plate}_" f"id_{track_id}.jpg"))
                                if cv2.imwrite(crop_path, best_crop):
                                    luuThongTinBienSo(dinhDangBienSo(final_plate), crop_path)
                                    saved_tracks.add(track_id)
                    result = track_results.get(track_id)
                    if result is not None:
                        color = (0, 255, 0)
                        plate_text = dinhDangBienSo(result["plate"])
                        label = (f"ID {track_id} | " f"{plate_text} ")
                    else:
                        color = (0, 165, 255)
                        history = ocr_history.get(track_id)
                        raw_text = ""
                        if history:
                            raw_text = history[-1].get("raw_text", "")
                        label = (f"ID {track_id} | " f"DET {detector_confidence:.2f}")
                        if raw_text:
                            label += (f" | OCR {raw_text}")
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_width = text_size[0]
                    text_height = text_size[1]
                    label_y = max(text_height + 12, y1)
                    cv2.rectangle(output_frame, (x1, label_y - text_height - 12), (min(video_width, x1 + text_width + 12), label_y + 5), color, -1)
                    cv2.putText(output_frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        expired_track_ids = [track_id for track_id, last_frame in last_seen_frame.items() if frame_count - last_frame > 150]
        for track_id in expired_track_ids:
            ocr_history.pop(track_id, None)
            track_results.pop(track_id, None)
            best_crops.pop(track_id, None)
            last_seen_frame.pop(track_id, None)
        current_time = time.perf_counter()
        elapsed_time = current_time - previous_time
        previous_time = current_time
        if elapsed_time > 0:
            current_fps = 1.0 / elapsed_time
            display_fps = display_fps * 0.9 + current_fps * 0.1
        cv2.putText(output_frame, f"FPS: {display_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output_frame, f"Frame: {frame_count}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(output_frame)
        display_frame = output_frame
        if video_width > 1280:
            display_width = 1280
            display_height = int(video_height * display_width / video_width)
            display_frame = cv2.resize(output_frame, (display_width, display_height))
        cv2.imshow("CAM", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.waitKey(0)
finally:
    video.release()
    writer.release()
    cv2.destroyAllWindows()
print("Đã xử lý:", frame_count, "frame")
print("Video kết quả:", OUTPUT_VIDEO_PATH)
print("Ảnh crop:", OUTPUT_CROP_DIR)