import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

app = FastAPI(title="face-crop-service", version="1.0.0")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


def resize_with_aspect_and_pad(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT_101,
    )


@app.post("/crop")
async def crop_face(
    file: UploadFile = File(...),
    size: int = Form(256),
    padding_x: float = Form(0.2),
    padding_y: float = Form(0.3),
    padding_left: float = Form(None),
    padding_right: float = Form(None),
    padding_top: float = Form(None),
    padding_bottom: float = Form(None),
):
    payload = await file.read()
    arr = np.frombuffer(payload, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid image")

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise HTTPException(status_code=422, detail="no face detected")

    face_landmarks = results.multi_face_landmarks[0]
    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = max(1, x_max - x_min)
    bh = max(1, y_max - y_min)
    left = padding_left if padding_left is not None else padding_x
    right = padding_right if padding_right is not None else padding_x
    top = padding_top if padding_top is not None else padding_y
    bottom = padding_bottom if padding_bottom is not None else padding_y

    x_min = max(0, x_min - int(bw * left))
    x_max = min(w, x_max + int(bw * right))
    y_min = max(0, y_min - int(bh * top))
    y_max = min(h, y_max + int(bh * bottom))

    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        raise HTTPException(status_code=422, detail="empty crop")

    crop = resize_with_aspect_and_pad(crop, size)
    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return Response(content=out.tobytes(), media_type="image/png")

