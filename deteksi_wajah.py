import os
import sys
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Fungsi untuk memuat model YOLO
def load_model(model_path):
    try:
        print(f"Memuat model dari {model_path}...")
        model = YOLO(model_path)
        print("Model berhasil dimuat!")
        return model
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {e}")
        sys.exit(1)

# Fungsi untuk melakukan deteksi pada gambar
def detect_image(image_path, model):
    try:
        print(f"Melakukan deteksi pada gambar: {image_path}")
        # Lakukan prediksi pada gambar
        results = model(image_path)
        # Visualisasikan hasil dengan menampilkan gambar yang sudah dideteksi
        print("Menampilkan hasil deteksi...")
        results.show()  # Menampilkan hasil deteksi menggunakan library bawaan YOLO
        results.save(save_dir="runs/detect")  # Menyimpan hasil deteksi ke folder
        # Menampilkan hasil dalam format gambar dengan matplotlib
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Hasil Deteksi")
        plt.axis('off')
        plt.show()
        print(f"Deteksi pada gambar {image_path} selesai.")
    except Exception as e:
        print(f"Terjadi kesalahan saat mendeteksi gambar: {e}")

# Fungsi untuk melakukan deteksi pada kamera langsung
def detect_live(model):
    try:
        print("Memulai deteksi kamera langsung...")
        cap = cv2.VideoCapture(0)  # Membuka kamera default
        if not cap.isOpened():
            print("Gagal membuka kamera. Pastikan kamera terhubung dengan benar.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Tidak dapat membaca frame dari kamera.")
                break
            # Lakukan prediksi pada frame
            results = model(frame)
            # Plot hasil deteksi dengan bounding boxes
            annotated_frame = results[0].plot()
            # Tampilkan frame dengan deteksi objek
            cv2.imshow("YOLO Deteksi Kamera Langsung", annotated_frame)

            # Jika pengguna menekan 'q', keluar dari kamera
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Keluar dari deteksi kamera langsung...")
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Terjadi kesalahan saat deteksi kamera langsung: {e}")

# Fungsi untuk validasi path model
def validate_model_path(model_path):
    if not os.path.exists(model_path):
        print(f"Path model {model_path} tidak ditemukan. Pastikan file model ada.")
        sys.exit(1)

# Fungsi utama untuk memilih mode deteksi
def main():
    print("Selamat datang di aplikasi deteksi objek menggunakan YOLOv8!")
    print("Pilih Mode Deteksi:")
    print("1. Deteksi pada Gambar")
    print("2. Deteksi pada Kamera Langsung")
    
    # Input pilihan pengguna
    choice = input("Masukkan pilihan Anda (1/2): ").strip()

    # Tentukan path model YOLO
    model_path = "yolo11.pt"  # Ganti dengan path model YOLO yang sesuai
    validate_model_path(model_path)
    
    # Muat model YOLO
    model = load_model(model_path)
    
    # Pilih mode deteksi
    if choice == "1":
        # Mode deteksi gambar
        image_path = input("Masukkan path gambar yang ingin dideteksi: ").strip()
        if os.path.exists(image_path):
            detect_image(image_path, model)
        else:
            print(f"Path gambar {image_path} tidak ditemukan.")
    elif choice == "2":
        # Mode deteksi kamera langsung
        detect_live(model)
    else:
        print("Pilihan tidak valid! Program akan keluar.")
        sys.exit(1)

# Menjalankan aplikasi
if __name__ == "__main__":
    main()