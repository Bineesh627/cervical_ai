# Cervical Cancer Detection with Federated Learning

## Overview

This project is a comprehensive **Multimodal AI System** designed for the early detection and risk assessment of cervical cancer. It combines **clinical data** (age, history, etc.) and **Pap smear images** to provide accurate predictions.

Key to this project is the integration of **Federated Learning (FL)**. This ensures patient privacy by training AI models locally on patient data without ever uploading sensitive raw images to a central training server.

## Features

- **Multimodal Analysis**: Combines tabular clinical data and image data using Fusion models for higher accuracy.
- **Federated Learning**: Uses [Flower (flwr)](https://flower.dev/) to enable privacy-preserving, distributed model training.
- **Role-Based Access**:
  - **Patients**: Can sign up, upload reports, view history, and ask questions.
  - **Doctors**: Can view patient risks, manage patients, and answer doubts.
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Visualizes which parts of the Pap smear image contributed to the prediction.
  - **SHAP**: Explains the impact of clinical features (features like age, smoking history) on the risk score.
- **Secure Dashboard**: User-friendly web interface built with Django.

## Technology Stack

- **Backend Framework**: Django (Python)
- **Machine Learning**: PyTorch, Torchvision
- **Federated Learning**: Flower (flwr)
- **Image Processing**: OpenCV, Pillow
- **Explainability**: SHAP, Grad-CAM (pytorch-gradcam)
- **Database**: SQLite (default) / PostgreSQL
- **Frontend**: HTML5, CSS3, JavaScript (Bootstrap)

## Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Bineesh627/cervical_ai.git
    cd cervical_ai
    ```

2.  **Set Up Virtual Environment**

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Database Migrations**
    ```bash
    python manage.py migrate
    ```

## How to Run

To fully utilize the system, you need to run the Federated Learning server and the Web application.

### 1. Start the Federated Learning Server

This server coordinates the global model updates. Open a terminal and run:

```bash
python fedrated/fed_server.py
```

_The server will listen on port 8091._

### 2. Start the Web Application

Open a **new terminal**, activate the environment, and run the Django development server:

```bash
python manage.py runserver
```

_Access the app at `http://127.0.0.1:8000/`._

## Usage Workflow

1.  **Register/Login**: Sign up as a Patient or Doctor.
2.  **Patient Flow**:
    - Navigate to **Upload Pap Smear**.
    - Fill in clinical details and upload an image.
    - Upon submission, the system provides an immediate prediction (High/Low risk).
    - **Behind the Scenes**: The application triggers a background process to "train" on this new data locally and connect to the FL Server (if running) to update the global model.
3.  **Doctor Flow**:
    - View a dashboard of high-risk patients.
    - Review detailed reports including Grad-CAM heatmaps and SHAP plots.
    - Reply to patient queries.
