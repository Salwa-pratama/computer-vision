from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# 1. Ini merupakan daftar objek yang bisa dideteksi oleh Mobilenet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 2. Load model ke OpenCV
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")


def generate_frames():
    camera = cv2.VideoCapture(0) # Gunakan 0 untuk webcam lokal

    while True:
        success, frame = camera.read()
        if not success:
            break

        # 3. Ubah ukuran frame agar ringan diproses (300x300 pixel)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # 4. Masukkan gambar ke model AI
        net.setInput(blob)
        detections = net.forward()

        # 5. Gambar kotak untuk setiap objek yang terdeteksi
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Hanya tampilkan jika AI yakin di atas 50%
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 6. Konversi gambar jadi format biner untuk dikirim ke HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




# Endpoint API
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Jalan di semua IP container agar bisa diakses browser
    app.run(host='0.0.0.0', port=5000)
