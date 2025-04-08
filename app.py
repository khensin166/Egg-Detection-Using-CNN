import os
import datetime
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = load_model('model/model_telur2.h5')
class_names = ['dead', 'fertile', 'infertile', 'not_egg']  # Perbaikan nama kelas

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400
    
    # Simpan file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocess gambar
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Timestamp untuk waktu analisis
    analysis_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    return render_template('index.html', 
                          prediction=predicted_class,
                          confidence=f"{confidence*100:.2f}%", 
                          image_path=filepath,
                          analysis_time=analysis_time)

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)