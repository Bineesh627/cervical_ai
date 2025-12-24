import os

# Define the project structure
structure = {
    "cervical_multimodal": {
        "manage.py": "",
        "requirements.txt": "",
        "Dockerfile": "",
        "docker-compose.yml": "",
        "README.md": "",
        "tmp": {},
        "data": {},
        "models": {
            "clinical_num_imputer.joblib": "",
            "clinical_scaler.joblib": "",
            "clinical_xgb.joblib": "",
            "image_cnn.pth": ""
        },
        "fedrated": {
            "fed_server.py": "",
            "fed_client.py": ""
        },
        "src": {
            "clinical_data_prep.py": "",
            "fusion.py": "",
            "gradcam.py": "",
            "shap_explain.py": "",
            "train_clinical.py": "",
            "train_image.py": "",
            "utils.py": ""
        },
        "cervical": {
            "__init__.py": "",
            "admin.py": "",
            "apps.py": "",
            "models.py": "",
            "forms.py": "",
            "views.py": "",
            "urls.py": "",
            "templates": {
                "cervical": {
                    "base.html": "",
                    "login.html": "",
                    "signup_patient.html": "",
                    "signup_doctor.html": "",
                    "patient_dashboard.html": "",
                    "clinical_form.html": "",
                    "image_upload.html": "",
                    "patient_detail.html": "",
                    "doctor_dashboard.html": ""
                }
            },
            "static": {
                "cervical": {
                    "css": {
                        "styles.css": ""
                    },
                    "uploads": {}
                }
            }
        },
        "project_settings": {
            "__init__.py": "",
            "settings.py": "",
            "urls.py": "",
            "wsgi.py": ""
        },
        "logs": {}
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)

# Run
create_structure(".", structure)
print("Project structure created successfully!")
