# ♻️ AI-Powered Waste Detection System

An AI-powered waste detection system built using **YOLOv8**, **Streamlit**, and **Ollama**. The application detects waste objects from images or live webcam input and provides AI-generated waste management recommendations using a locally deployed Large Language Model (LLM).

---

## 📌 Project Overview

Proper waste management plays a vital role in protecting the environment and promoting sustainability. This project uses a custom-trained **YOLOv8** model to detect waste objects and integrates **Ollama** to generate intelligent disposal recommendations locally without relying on cloud-based AI services.

The application offers a simple and interactive Streamlit interface where users can upload an image or use a live webcam for real-time waste detection and AI-powered analysis.

---

## ✨ Current Features

- 🔍 Waste Detection using YOLOv8
- 🖼️ Image Upload Detection
- 📷 Live Webcam Detection
- 🤖 AI-generated Waste Management Recommendations
- 🧠 Local LLM Integration using Ollama
- ⚡ Real-time Detection Results
- 🎨 Interactive Streamlit User Interface
- 🌐 Public Deployment using ngrok

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| YOLOv8 | Waste Detection |
| Streamlit | Web Application |
| OpenCV | Image Processing |
| Ultralytics | Object Detection |
| Ollama | Local LLM Runtime |
| Llama 3.2 | AI Recommendation |
| ngrok | Public Deployment |

---

## 📂 Project Structure

```
AI-Waste-Detection-and-Segregation/
│
├── app.py
├── helper.py
├── requirements.txt
├── README.md
├── weights/
│   └── best.pt
├── utils/
└── assets/
```

---

## ⚙️ Workflow

```
Image / Webcam
        │
        ▼
YOLOv8 Waste Detection
        │
        ▼
Detected Waste Objects
        │
        ▼
Ollama (Local LLM)
        │
        ▼
AI-generated Waste Recommendation
        │
        ▼
Streamlit Interface
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chatanushkac/AI-Waste-Detection-and-Segregation.git
```

### 2. Navigate to the Project Folder

```bash
cd AI-Waste-Detection-and-Segregation
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 💡 Current Capabilities

- Detects waste objects using a custom-trained YOLOv8 model.
- Supports both image upload and live webcam detection.
- Generates AI-powered waste management recommendations using a locally deployed LLM.
- Eliminates dependency on cloud-based AI APIs by integrating Ollama.
- Can be accessed remotely through ngrok for demonstration purposes.

---

## 🚧 Future Enhancements

The following features are planned for future development:

- ♻️ Waste Categorization
  - Compostable Waste
  - Recyclable Waste
  - E-Waste
  - General Waste

- 🗑️ Smart Bin Recommendation

- 📊 Waste Analytics Dashboard

- 🌱 Carbon Footprint Estimation

- 📱 Mobile Application Support

- ☁️ Cloud Database Integration

- 🌍 Multi-language Support

---

## 👩‍💻 Developer

**Anushka Chaturvedi** **&**
**Hanumant Sahay**

B.Tech CSE (AI & Data Science)

---

## 📄 License

This project is developed for educational and internship purposes.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
