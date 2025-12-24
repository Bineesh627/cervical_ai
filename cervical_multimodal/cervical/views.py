import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.db.models import Prefetch, Max
# Import Django's messaging system for user feedback
from django.contrib import messages 

from src.predict_wrappers import (
    multimodal_predict, 
    clinical_predict, 
    image_predict, 
    fuse_probs
)
from fedrated.fed_client import train_locally
import threading

from .forms import (
    PatientSignUpForm, DoctorSignUpForm, ClinicalForm, 
    PapImageForm, PatientProfileUpdateForm, DoctorNewPatientForm
) 

from .models import PatientProfile, DoctorProfile, PatientRecord, PatientDoubt, User 


# ---------------- HOME ----------------
def home_redirect(request):
    """Redirects the base URL to the main authentication container."""
    return redirect('auth_container')

# ---------------- AUTH ----------------
def auth_container(request):
    """
    Renders the unified login/signup container page. 
    This is the default landing view.
    """
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': PatientSignUpForm(),
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'login'  # Default to showing the Login panel
    }
    return render(request, 'cervical/auth_container.html', context)


def signup_patient(request):
    """Handles Patient signup - redirects to login after successful signup."""
    if request.user.is_authenticated:
        logout(request)

    form = PatientSignUpForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.save() 
            PatientProfile.objects.get_or_create(user=user)
            messages.success(request, "Signup successful! Please log in.")
            return redirect('auth_container')
        else:
            print("\n--- PATIENT SIGNUP FAILED ---")
            print(form.errors)
            print("-----------------------------\n")
    
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': form,
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'signup_patient'
    }
    return render(request, 'cervical/auth_container.html', context)


def signup_doctor(request):
    """Handles Doctor signup - redirects to login after successful signup."""
    if request.user.is_authenticated:
        logout(request)

    form = DoctorSignUpForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.save()
            DoctorProfile.objects.get_or_create(user=user)
            messages.success(request, "Signup successful! Please log in.")
            return redirect('auth_container')
        else:
            print("\n--- DOCTOR SIGNUP FAILED ---")
            print(form.errors)
            print("----------------------------\n")
            
    context = {
        'login_form': AuthenticationForm(),
        'patient_form': PatientSignUpForm(),
        'doctor_form': form,
        'current_view': 'signup_doctor'
    }
    return render(request, 'cervical/auth_container.html', context)


def user_login(request):
    """Handles user login."""
    form = AuthenticationForm(request, data=request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            if user.role == 'patient':
                return redirect('patient_dashboard') 
            elif user.role == 'doctor':
                return redirect('doctor_dashboard')
            else:
                logout(request)
                return redirect('auth_container')

        else:
            print("\n--- LOGIN FAILED ---")
            print(form.errors)
            print("--------------------\n")
            messages.error(request, "Invalid login credentials.")

    context = {
        'login_form': form,
        'patient_form': PatientSignUpForm(),
        'doctor_form': DoctorSignUpForm(),
        'current_view': 'login'
    }
    return render(request, 'cervical/auth_container.html', context)


def user_logout(request):
    """Logs the user out."""
    logout(request)
    return redirect('auth_container')


# ---------------- PATIENT ----------------
@login_required
def patient_dashboard(request):
    """Displays patient's profile and test history."""
    if request.user.role != 'patient':
        # Fallback in case a doctor ends up here
        return redirect('doctor_dashboard')
    
    profile = get_object_or_404(PatientProfile, user=request.user)
    # Prefetch the doubts for each record for efficient rendering
    records = PatientRecord.objects.filter(patient=profile).prefetch_related('doubts').order_by('-created_at')
    
    return render(request, 'cervical/patient_dashboard.html', {
        'profile': profile,
        'records': records
    })

from decimal import Decimal, InvalidOperation

@login_required
def clinical_entry(request):
    """Handles clinical data entry and prediction (without image)."""
    print("Clinical entry accessed by:", request.user.role)
    profile = get_object_or_404(PatientProfile, user=request.user)

    form = ClinicalForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        rec = form.save(commit=False)
        rec.patient = profile
        rec.save()

        features = {
            'age': rec.age or 0,
            'hpv_result': rec.hpv_result,
            'smoking': rec.smoking_years,
            'contraception': rec.contraception_years,
            'sexual_history': rec.sexual_partners,
        }
        print(rec.id, "Clinical features:", features)

        # --- Predict (SHAP must be non-fatal) ---
        try:
            score, _label_from_model, shap_path = clinical_predict(features, record_id=rec.id)
        except Exception as e:
            print(f"[Warning] clinical_predict failed (non-fatal): {e}")
            score, shap_path = 0.0, ""

        try:
            score_dec = Decimal(str(score))        # preserve exact text form
        except InvalidOperation:
            score_dec = Decimal("0")

        rec.clinical_risk_score = score_dec
        rec.clinical_pred_label = "High" if float(score_dec) >= 0.005 else "Low"
        rec.save()

        messages.success(
            request,
            "Clinical analysis complete." + (" SHAP added." if shap_path else " (explainability skipped)")
        )
        return redirect('patient_dashboard')

    return render(request, 'cervical/clinical_form.html', {'form': form})


@login_required
def upload_pap(request):
    """Handles image upload and multimodal prediction."""
    profile = get_object_or_404(PatientProfile, user=request.user)
    form = PapImageForm(request.POST or None, request.FILES or None)

    if request.method == 'POST' and form.is_valid():
        rec = form.save(commit=False)
        rec.patient = profile
        rec.save()

        features = {
            'age': rec.age or 0,
            'hpv_result': rec.hpv_result,
            'smoking': rec.smoking_years,
            'contraception': rec.contraception_years,
            'sexual_history': rec.sexual_partners,
        }

        result = multimodal_predict(rec.image.path, features, rec.id)

def clean_path(full_path):
    """
    Converts a full file system path to a URL-relative path expected by {% static %}.
    Finds the first instance of 'cervical/static/' or 'static/' and strips everything before it.
    """
    if not full_path:
        return ""
    
    # Normalize and convert to Path object for reliable splitting
    p = Path(full_path).as_posix()
    
    # --- DEBUGGING STEP ---
    print(f"\n--- PATH DEBUG ---")
    print(f"Input Path: {p}")
    # ----------------------
    
    static_prefix = 'cervical/static/'
    if static_prefix in p:
        relative_path = p.split(static_prefix, 1)[-1]
    
    # 2. If that fails, try finding 'static/' (7 chars)
    elif 'static/' in p:
        relative_path = p.split('static/', 1)[-1]
        
    else:
        # 3. If no recognizable prefix is found, save the input path
        relative_path = p
        
    # --- DEBUGGING STEP ---
    print(f"Saved Path: {relative_path}")
    print(f"-------------------\n")
    # ----------------------

    return relative_path

from pathlib import Path

@login_required
def upload_pap(request):
    profile = get_object_or_404(PatientProfile, user=request.user)
    form = PapImageForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        if form.is_valid():
            rec = form.save(commit=False)
            rec.patient = profile
            rec.save()

            features = {
                'age': rec.age or 0,
                'hpv_result': rec.hpv_result,
                'smoking': rec.smoking_years,
                'contraception': rec.contraception_years,
                'sexual_history': rec.sexual_partners,
            }

            print(f"\n{'='*60}")
            print(f"Starting multimodal prediction for record ID: {rec.id}")
            print(f"Image path: {rec.image.path}")
            print(f"Features: {features}")
            print(f"{'='*60}\n")

            result = multimodal_predict(rec.image.path, features, rec.id)

            print(f"\n{'='*60}")
            print(f"Multimodal prediction results:")
            print(f"  clinical_prob: {result.get('clinical_prob')} (type: {type(result.get('clinical_prob'))})")
            print(f"  image_prob: {result.get('image_prob')} (type: {type(result.get('image_prob'))})")
            print(f"  fused_score: {result.get('fused_score')} (type: {type(result.get('fused_score'))})")
            print(f"  gradcam_path: {result.get('gradcam_path')}")
            print(f"  shap_path: {result.get('shap_path')}")
            print(f"{'='*60}\n")

            # --- scores ---
            rec.clinical_risk_score = float(result.get("clinical_prob", 0.0))
            rec.image_prob  = float(result.get("image_prob", 0.0))
            rec.fused_score         = float(result.get("fused_score", 0.0))

            # --- labels strictly from the scores (0.5) ---
            rec.clinical_pred_label = "High" if rec.clinical_risk_score >= 0.005 else "Low"
            rec.image_label = result.get("image_label") or ("High" if rec.image_prob >= 0.005 else "Low")
            rec.fused_label         = "High" if rec.fused_score         >= 0.005 else "Low"

            # --- paths (relative for {% static %}) ---
            rec.gradcam_path        = clean_path(result.get("gradcam_path") or "")
            rec.clinical_shap_path  = clean_path(result.get("shap_path") or "")

            print(f"\n{'='*60}")
            print(f"Saving record with values:")
            print(f"  clinical_risk_score: {rec.clinical_risk_score}")
            print(f"  image_prob: {rec.image_prob}")
            print(f"  fused_score: {rec.fused_score}")
            print(f"  gradcam_path: {rec.gradcam_path}")
            print(f"  clinical_shap_path: {rec.clinical_shap_path}")
            print(f"{'='*60}\n")

            rec.save()

            # --- Federated Learning Trigger ---
            try:
                # Run FL client in a separate thread to avoid blocking the user response
                t = threading.Thread(target=train_locally, args=(rec.patient.id,))
                t.daemon = True
                t.start()
                print(f"Triggered FL training for patient {rec.patient.id}")
            except Exception as e:
                print(f"Failed to trigger FL: {e}")
            # ----------------------------------

            messages.success(request, "Pap image uploaded and analysis complete.")
            return redirect('patient_dashboard')
        else:
            print("FORM IS INVALID. Errors:", form.errors)
            messages.error(request, "Image upload failed. Please check form errors.")

    return render(request, 'cervical/image_upload.html', {'form': form})


@login_required
def patient_detail(request, record_id):
    """
    Displays details for a specific patient record and handles form submissions 
    for asking doubts and referring to a doctor.
    """
    rec = get_object_or_404(PatientRecord, id=record_id)
    
    # Security check: Patient can only view their own records
    if request.user.role == 'patient' and rec.patient.user != request.user:
        messages.error(request, "Access denied.")
        return redirect('patient_dashboard')
        
    if request.method == 'POST' and request.user.role == 'patient':
        action = request.POST.get('action')
        msg = request.POST.get('message', '').strip()

        if action == 'ask_doubt' and msg:
            # 🔑 FIX: Use 'question=msg' to match the PatientDoubt model field
            PatientDoubt.objects.create(
                record=rec, 
                sender=request.user, 
                question=msg  
            )
            messages.success(request, "Your question has been sent to the doctor.")
            
        elif action == 'refer':
            # Note: The model uses referral_status='R', not a boolean 'referred'
            rec.referral_status = 'R'
            rec.save()
            messages.info(request, "This record has been flagged for doctor consultation.")
            
        # Redirect after POST to prevent resubmission
        return redirect('patient_detail', record_id=rec.id)
    
    # Retrieve all doubts related to this record
    doubts = PatientDoubt.objects.filter(record=rec).order_by('-created_at')
            
    return render(request, 'cervical/patient_detail.html', {'rec': rec, 'doubts': doubts})


@login_required
def update_patient_profile(request):
    """Handles updating a patient's profile details."""
    profile = get_object_or_404(PatientProfile, user=request.user)
    
    if request.method == 'POST':
        form = PatientProfileUpdateForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated successfully.")
            return redirect('patient_dashboard')
    else:
        form = PatientProfileUpdateForm(instance=profile)
        
    return render(request, 'cervical/patient_update.html', {'form': form})

@login_required
def ask_doubt_view(request):
    """
    Handles POST requests from the 'Ask Doubt' form. 
    It is used only to capture the form data and redirect, 
    as the main logic is handled in patient_detail. 
    
    NOTE: This view is necessary only because it is referenced in urls.py.
    """
    if request.user.role != 'patient':
        messages.error(request, "Access denied.")
        return redirect('auth_container')
        
    # Check for POST data submitted by the doubt form
    if request.method == 'POST':
        # The form should include a hidden input for the record ID
        record_id = request.POST.get('record_id') 
        
        # If record_id is present, redirect to the detail view to process the POST request
        # The detail view's logic will handle the actual creation of the PatientDoubt object.
        if record_id:
            # Re-submit the POST data to the patient_detail view
            return redirect('patient_detail', record_id=record_id)
        else:
            messages.error(request, "Missing record ID for doubt submission.")

    # Fallback redirect to the main patient dashboard
    return redirect('patient_dashboard')


import os
from pathlib import Path  # Required for clean_path
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.db.models import Max, Subquery, OuterRef
from django.utils import timezone

# Assuming your imports are defined here:
# from .models import PatientProfile, PatientRecord, PatientDoubt, User 
# from .forms import PapImageForm, DoctorNewPatientForm 
# from src.predict_wrappers import multimodal_predict # Assuming this is your wrapper

# Helper function to restrict access to users with the 'doctor' role
def is_doctor(user):
    return user.is_authenticated and user.role == 'doctor'

# --- FILE PATH CLEANUP FUNCTION (Crucial for image display) ---
def clean_path(full_path):
    """
    Converts a full file system path to a URL-relative path expected by {% static %}.
    Finds the path segment after the project's 'cervical/static/'.
    """
    if not full_path:
        return ""
    
    # Normalize and convert to Path object for reliable splitting
    p = Path(full_path).as_posix()
    
    static_prefix = 'cervical/static/'
    
    # Check for the specific static prefix
    if static_prefix in p:
        # Return everything after the prefix
        return p.split(static_prefix, 1)[-1]
    
    # Fallback to just the path if the prefix isn't found (though this shouldn't happen)
    return p

# ---------------------------------------------------------------


# ---------------- DOCTOR VIEWS ----------------
@login_required
@user_passes_test(is_doctor)
def doctor_dashboard(request):
    """Displays doctor dashboard summary and list of patients/messages."""
    # Ensure the doctor check is done via decorator, but keep the redirect fallback
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
    
    total_patients = PatientProfile.objects.all().count()
    
    # --- FIX 1: Simplified and corrected logic for fetching the LATEST record per patient ---
    latest_records_qs = PatientRecord.objects.filter(patient=OuterRef('id')).order_by('-created_at')
    
    # Filter PatientProfiles that have a PatientRecord whose latest entry is 'High'
    # This uses a subquery to find the fused_label of the latest record
    high_risk_patients = PatientProfile.objects.annotate(
        latest_fused_label=Subquery(latest_records_qs.values('fused_label')[:1])
    ).filter(latest_fused_label='High')

    high_risk_count = high_risk_patients.count()
    
    # Total tests count is better fetched from all records
    total_tests = PatientRecord.objects.count()

    # Messages where is_answered is False
    unanswered_messages = PatientDoubt.objects.filter(is_answered=False).count() 
    
    # Fetch patients and their records (ordered by latest) for the list display
    patients = PatientProfile.objects.all().prefetch_related(
        Prefetch('records', queryset=PatientRecord.objects.order_by('-created_at'))
    ).order_by('user__last_name')
    
    messages_list = PatientDoubt.objects.filter(is_answered=False).select_related('record__patient__user').order_by('-created_at') 

    context = {
        'total_patients': total_patients,
        'high_risk_count': high_risk_count,
        'unanswered_messages': unanswered_messages,
        'total_tests': total_tests, # Added total tests context
        'patients': patients, 
        'messages': messages_list,
    }

    return render(request, 'cervical/doctor_dashboard.html', context)


@login_required
@user_passes_test(is_doctor)
def doctor_predict(request):
    """Doctor interface to select/create a patient and input data for multimodal prediction."""
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
    
    all_patients = PatientProfile.objects.all().select_related('user').order_by('user__last_name', 'user__first_name')
    
    new_patient_form = DoctorNewPatientForm(request.POST or None, prefix='new')
    prediction_form = PapImageForm(request.POST or None, request.FILES or None, prefix='predict')

    if request.method == 'POST':
        patient_profile = None

        # --- 1. Identify/Create Patient ---
        # Note: Added check for 'new-email' which indicates the New Patient form was likely submitted
        if 'new-email' in request.POST and new_patient_form.is_valid():
            # ... Patient creation logic remains the same ...
            email = new_patient_form.cleaned_data['email']
            user, created = User.objects.get_or_create(email=email, defaults={
                'username': email, 
                'first_name': new_patient_form.cleaned_data['first_name'],
                'last_name': new_patient_form.cleaned_data['last_name'],
                'role': 'patient',
                'is_active': True
            })
            if created:
                user.set_unusable_password() 
                user.save()
            
            patient_profile, _ = PatientProfile.objects.get_or_create(
                user=user, 
                defaults={
                    'age': new_patient_form.cleaned_data.get('age', 0),
                    'sex': new_patient_form.cleaned_data.get('sex', 'U'),
                    'blood_group': new_patient_form.cleaned_data.get('blood_group', 'U'),
                }
            )
            messages.success(request, f"Patient {user.email} selected/created successfully.")
            
        elif request.POST.get('patient_select'):
            patient_id = request.POST.get('patient_select')
            patient_profile = get_object_or_404(PatientProfile, id=patient_id)
        
        # --- 2. Run Prediction ---
        if patient_profile:
            # Re-validate the prediction form after determining the patient
            if prediction_form.is_valid():
                rec = prediction_form.save(commit=False)
                rec.patient = patient_profile
                rec.save() 
                
                img_path = rec.image.path
                features = {
                    'age': rec.age or 0,
                    'hpv_result': rec.hpv_result,
                    'smoking': rec.smoking_years,
                    'contraception': rec.contraception_years,
                    'sexual_history': rec.sexual_partners,
                }
                
                # Assuming multimodal_predict is available
                result = multimodal_predict(img_path, features, rec.id)

                rec.clinical_risk_score = result.get("clinical_prob")
                rec.clinical_pred_label = result.get("clinical_label")
                rec.image_prob = result.get("image_prob")
                rec.image_label = result.get("image_label")
                rec.fused_score = result.get("fused_score")
                rec.fused_label = result.get("fused_label")
                
                # --- FIX 2: Apply clean_path here! ---
                rec.gradcam_path = clean_path(result.get("gradcam_path"))
                rec.clinical_shap_path = clean_path(result.get("shap_path")) 

                rec.save()
                messages.success(request, f"Prediction complete for {patient_profile.user.email}.")
                return redirect('doctor_view_patient_record', record_id=rec.id)
            else:
                # If prediction form fails, re-render with errors
                messages.error(request, "Prediction form validation failed. Please correct the fields.")
                
        else:
            messages.error(request, "Patient was not selected or created correctly. Please check all forms.")

    context = {
        'patients': all_patients,
        'prediction_form': prediction_form,
        'new_patient_form': new_patient_form,
    }
    return render(request, 'cervical/doctor_predict.html', context)


@login_required
@user_passes_test(is_doctor)
def doctor_view_patient_record(request, record_id):
    """Doctor view of patient record and message reply handling."""
    # if request.user.role != 'doctor':
    #     return redirect('patient_dashboard')
        
    rec = get_object_or_404(PatientRecord.objects.select_related('patient__user'), id=record_id)
    
    if request.method == 'POST':
        if 'reply_message' in request.POST:
            msg_id = request.POST.get('message_id')
            reply_text = request.POST.get('reply_text', '').strip()
            
            if reply_text:
                try:
                    msg = PatientDoubt.objects.get(id=msg_id) 
                    msg.answer = reply_text
                    msg.is_answered = True
                    msg.answered_at = timezone.now()
                    msg.save()
                    messages.success(request, "Response sent successfully.")
                except PatientDoubt.DoesNotExist:
                    messages.error(request, "Message not found.")
            else:
                messages.warning(request, "Reply text cannot be empty.")
            
    # Retrieve all doubts related to this record
    doubts = PatientDoubt.objects.filter(record=rec).order_by('-created_at')
            
    return render(request, 'cervical/patient_detail.html', {'rec': rec, 'doubts': doubts})
