from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# 1. Ganti array CLASSES jadi Dictionary khusus untuk alat belajar
# Di dataset COCO: 1 = Orang, 73 = Laptop, 77 = HP, 84 = Buku
COCO_LABELS = {
    1: "Person (User)",
    73: "Laptop",
    77: "Cell Phone (Distraction!)",
    84: "Book"
}
# 2. Load model versi TensorFlow (bukan Caffe lagi)
net = cv2.dnn.readNetFromTensorflow(
    "./model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
    "./model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
)

def generate_frames():
    camera = cv2.VideoCapture(0) # Gunakan 0 untuk webcam lokal

    while True:
        success, frame = camera.read()
        if not success:
            break

        # 3. Ubah ukuran frame agar ringan diproses (300x300 pixel)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), swapRB=True, crop=False)

        # 4. Masukkan gambar ke model AI
        net.setInput(blob)
        detections = net.forward()

       # 5. Gambar kotak HANYA untuk objek belajar & distraksi
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Hanya tampilkan jika AI yakin di atas 40%
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])

                # Cek apakah objek yang terdeteksi ada di daftar asisten belajar kita
                if idx in COCO_LABELS:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Buat warna kotak merah jika HP (distraksi), hijau jika lainnya
                    color = (0, 0, 255) if idx == 77 else (0, 255, 0)

                    label = f"{COCO_LABELS[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
