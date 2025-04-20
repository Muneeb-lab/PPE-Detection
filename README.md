
# 🦺 AI-Based PPE Compliance Detection System

This project detects whether industrial workers are wearing necessary Personal Protective Equipment (PPE) — such as helmets, gloves, and goggles — using a custom-trained YOLOv8 model. The system uses Streamlit for a user-friendly web interface and is deployed on Hugging Face Spaces for public access.

---

## 📌 Features

- Detects multiple people and checks helmet compliance
- Highlights non-compliant individuals
- Displays annotated image with bounding boxes
- Provides a summary of detected persons and their compliance
- Option to generate and download a CSV report
- Fully deployed on Hugging Face Spaces

---

## 🚀 Demo

🔗 [Live Demo on Hugging Face](https://huggingface.co/spaces/Muneeb-Ullah/MM)

📸 _(Insert screenshots here: Upload image showing detection results + Hugging Face interface)_

---

## 🧠 Model Training Summary

- **Framework**: Ultralytics YOLOv8
- **Dataset**: Custom-labeled PPE dataset (helmets, gloves, goggles, etc.)
- **Model**: YOLOv8m
- **Training Environment**: Google Colab
- **Performance**: _(Add metrics like mAP if available)_

📁 Trained model file: `ppe_detector.pt` (uploaded to Hugging Face Space)

---

## 📦 Folder Structure

```
ppe-detection-streamlit/
│
├── app.py                 # Streamlit app
├── ppe_detector.pt        # Trained YOLOv8 model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/ppe-detection-streamlit.git
cd ppe-detection-streamlit
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🔍 Dependencies

- streamlit  
- torch  
- ultralytics  
- opencv-python  
- numpy  
- Pillow  
- matplotlib  

All dependencies are listed in `requirements.txt`

---

## 📊 How It Works

1. Upload an image with workers
2. Model detects people and checks for helmet use
3. Bounding boxes are drawn with compliance status
4. Generate a timestamped CSV report if desired

📸 _(Insert screenshot of Streamlit app with bounding boxes here)_

---

## 💻 Deployment

- The project is deployed using **Hugging Face Spaces** with Streamlit interface
- Public access without authentication
- Ideal for quick demos and presentations

---

## 📚 Future Work

- Add detection for gloves, goggles, vests
- Integrate live video stream support
- Extend for real-time industrial use
- Track user-wise PPE violations

---

## 📬 Contact

👤 **Muneeb Ullah**  
📧 muneeb@example.com  
🔗 [LinkedIn](#) | [GitHub](https://github.com/your-username)

---

## 🏆 Credits

This project was developed as part of a Data Science Bootcamp final project, combining skills in machine learning, computer vision, and automation engineering.
