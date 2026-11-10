# 🇹🇭 Thai Fake News & Scam Detection

ระบบตรวจจับข่าวปลอมและข้อความหลอกลวงภาษาไทย โดยประยุกต์ใช้เทคโนโลยีประมวลผลภาษาธรรมชาติ (**Natural Language Processing: NLP**) ร่วมกับ **Machine Learning / Deep Learning** เพื่อช่วยคัดกรองและวิเคราะห์ความน่าเชื่อถือของเนื้อหาภาษาไทยได้อย่างมีประสิทธิภาพ

---

## ✨ Features (คุณสมบัติเด่น)

* **Thai Text Classification:** โมเดลจำแนกข้อความภาษาไทย (ข่าวปลอม / ข่าวจริง / ข้อความสแปมหลอกลวง)
* **Text Preprocessing:** ระบบคลีนข้อความ ตัดคำ และจัดการ Stopwords ภาษาไทยโดยเฉพาะ
* **Interactive Web UI:** หน้าเว็บอินเทอร์แอคทีฟสำหรับทดสอบป้อนข้อความและแสดงผลการทำนายแบบ Real-time
* **Confidence Score:** แสดงระดับความมั่นใจ (Confidence Rate %) ในการทำนายของโมเดล

---

## 🛠️ Tech Stack (เทคโนโลยีที่ใช้)

* **Language:** Python
* **NLP & Text Processing:** PyThaiNLP, Transformers / Hugging Face
* **Machine Learning & Modeling:** PyTorch / TensorFlow, Scikit-learn
* **Web Application:** Streamlit
* **Data & Visualization:** Pandas, NumPy, Plotly

---

## 🚀 Getting Started (วิธีการติดตั้งและเริ่มต้นใช้งาน)

### Prerequisites

* Python 3.9 ขึ้นไป
* pip (Python Package Installer)

### Installation & Run

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/necromancerth/real-or-cake.git](https://github.com/necromancerth/real-or-cake.git)
   cd real-or-cake


   Project Structure (โครงสร้างโปรเจกต์)

   ├── assets/             # รูปภาพและสไตล์สำหรับแสดงผลบน UI
├── data/               # สภาพแวดล้อมข้อมูลและชุดข้อมูลทดสอบ (Datasets)
├── models/             # โมเดลที่เทรนแล้ว (Trained Model Weights / Checkpoints)
├── src/
│   ├── preprocessing.py  # ฟังก์ชันทำความสะอาดและเตรียมข้อความภาษาไทย
│   ├── predict.py        # Pipeline การนำเข้าโมเดลและการประมวลผลคำตอบ
│   └── utils.py          # เครื่องมือเสริมอื่นๆ
├── app.py              # ไฟล์หลักสำหรับรัน Streamlit Web App
├── requirements.txt    # รายการ Library ที่จำเป็นต้องใช้
└── README.md
