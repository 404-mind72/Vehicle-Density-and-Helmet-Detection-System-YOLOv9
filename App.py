import cv2
import os
from ultralytics import YOLO
import winsound 


model_path = "9v-m/weights/best.pt"
model = YOLO(model_path)


video_path = "video/video.mp4"
cap = cv2.VideoCapture(video_path) 


output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# Mendapatkan properti video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Menginisialisasi video writer
output_path = os.path.join(output_dir, "output_video.mp4")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Label kelas
class_labels = ["bus", "mobil", "motor", "pakai_helm", "tidak_pakai_helm", "truck"]

# Memproses video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Menjalankan inferensi YOLO
    results = model(frame)

    # Menghitung deteksi
    class_counts = {label: 0 for label in class_labels}

    # Memproses deteksi
    for detection in results[0].boxes:
        cls_id = int(detection.cls)
        if 0 <= cls_id < len(class_labels):
            class_name = class_labels[cls_id]
            class_counts[class_name] += 1

            # Memeriksa jika 'tidak_pakai_helm' terdeteksi dan memberi suara beep
            if class_name == "tidak_pakai_helm" and class_counts[class_name] > 0:
                winsound.Beep(1000, 500)  # Memainkan suara beep dengan frekuensi 1000 Hz selama 500 ms

    # Menghitung total kepadatan untuk kelas yang relevan
    relevant_classes = ["bus", "mobil", "motor", "truck"]
    total_density = sum(class_counts[class_name] for class_name in relevant_classes)

    # Menentukan level kepadatan
    if total_density > 15:
        density_level = "Padat"
    elif total_density >= 10:
        density_level = "Sedang"
    else:
        density_level = "Rendah"

    # Visualisasi deteksi
    annotated_frame = results[0].plot(line_width=2, font_size=0.12)

    # Menambahkan teks jumlah deteksi dan level kepadatan
    y_offset = 20
    for class_name, count in class_counts.items():
        cv2.putText(annotated_frame, f"{class_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30

    # Menampilkan level kepadatan
    cv2.putText(annotated_frame, f"Total Kepadatan : {total_density}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y_offset += 30
    cv2.putText(annotated_frame, f"Level Kepadatan : {density_level}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Menulis frame ke output
    out.write(annotated_frame)

    # Mengubah ukuran frame untuk ditampilkan (opsional)
    resized_frame = cv2.resize(annotated_frame, (1280, 720))  # Mengatur ukuran menjadi 640x360

    # Menampilkan frame
    cv2.imshow("YOLOv8 Inference", resized_frame)

    # Menambahkan delay agar kecepatan video tetap normal
    # Menunggu sesuai dengan frame rate (fps)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Melepaskan sumber daya
cap.release()
out.release()
cv2.destroyAllWindows()