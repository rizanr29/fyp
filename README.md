# Final Year Project (FYP) – Human Pose Estimation & Performance Analysis
This project is developed as part of a Final Year Project to track and analyze soccer juggling performance using YOLOv8 Pose Estimation and Python multiprocessing to improve processing efficiency.
This repository contains the source code, documentation, and resources for my Final Year Project. The project focuses on human pose estimation using the YOLOv8 model and analyzes performance metrics based on movement tracking.

## 📁 Project Structure

- `main_mp.py`: Main script utilizing multiprocessing for video frame analysis.
- `dashboard.py`: Visual dashboard to display and interpret results.
- `utils_mp.py`: Utility functions designed to work efficiently with multiple processes.
- `data.py`, `guide.py`, `person.py`: Supporting Python modules.
- `Documents/`: Contains the final report and technical paper.
- `yolov8n.engine`, `yolov8n-pose.engine`: Optimized engine files for YOLO inference.

## 🚀 Features

- Real-time human pose detection
- Metric tracking for physical activities
- Exportable performance data for further analysis
- Visual dashboard for reporting

## 📄 Documents

Final project documentation can be found in the `Documents/` folder:
- Final Year Project Report (PDF)
- Technical Paper (PDF)

## ⚙️ Technologies Used
- Ultralytics YOLOv8 – Pose estimation model

- TensorRT – For optimized inference with .engine files

- PyTorch – Model backend and tensor computations

- Multiprocessing – Parallel processing of frames

- OpenCV – Video input/output and image operations

- Streamlit – Interactive performance dashboard

- Plotly – For dynamic graphs and charts

- Pandas / NumPy – Data manipulation and numerical analysis

- Google Sheets API (gspread + oauth2client / google-auth) – Logging session data

- Pynvml / psutil – Monitoring GPU/CPU usage

## 🔒 Sensitive Files

Some files (e.g., large model weights, keys) are excluded or removed due to size/security concerns. Refer to `.gitignore` and GitHub push protection rules.

## 📜 License

This project is for academic purposes and not intended for commercial use. Contact the author for more information.

---
