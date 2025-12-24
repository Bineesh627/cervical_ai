# cervical/urls.py (UPDATED)

from django.urls import path
from .import views

urlpatterns = [
    # Auth & home
    path('', views.home_redirect, name='home_redirect'),
    path('auth/', views.auth_container, name='auth_container'),

    path('home/', views.home_redirect, name='home_redirect'),
    path('signup/patient/', views.signup_patient, name='signup_patient'),
    path('signup/doctor/', views.signup_doctor, name='signup_doctor'),
    
    # FIX: Rename the URL name to 'user_login' for consistency
    path('login/', views.user_login, name='user_login'), 
    
    path('logout/', views.user_logout, name='logout'),
    path('patient/profile/update/', views.update_patient_profile, name='update_patient_profile'),


#path('doctor/messages/', views.doctor_messages_view, name='doctor_messages'),

    # Existing record view path (needed for the reply form action)
    path('doctor/record/<int:record_id>/', views.doctor_view_patient_record, name='doctor_view_patient_record'),





    # Patient
    path('patient/dashboard/', views.patient_dashboard, name='patient_dashboard'),
    path('patient/clinical/', views.clinical_entry, name='clinical_entry'),
    path('patient/upload/', views.upload_pap, name='upload_pap'),
    path('patient/record/<int:record_id>/', views.patient_detail, name='patient_detail'),

    # Doctor
    path('doctor/dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('doctor/predict/', views.doctor_predict, name='doctor_predict'),
    path('doctor/record/<int:record_id>/', views.doctor_view_patient_record, name='doctor_view_patient_record'),

    path('patient/doubt/', views.ask_doubt_view, name='ask_doubt'), 
]