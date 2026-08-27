package com.example.sgpc_app.data

import androidx.annotation.DrawableRes
import com.example.sgpc_app.R

data class Doctor(
    val id: Int,
    val name: String,
    val specialty: String,
    @DrawableRes val imageRes: Int,
    val rating: Int,
    val category: DoctorCategory,
    val patients: String = "1.2K",
    val experience: String = "8 yr",
    val ratingValue: String = "4.2",
    val description: String = ""
)

enum class DoctorCategory(val displayName: String) {
    PSYCHOLOGY("Psychology"),
    CARDIOLOGY("Cardiology"),
    DENTIST("Dentist")
}

fun getDoctorsByCategory(category: DoctorCategory): List<Doctor> {
    return allDoctors.filter { it.category == category }
}

val allDoctors = listOf(
    Doctor(
        id = 0,
        name = "Dr. Zainab Saeed",
        specialty = "Clinical Psychology",
        imageRes = R.drawable.doctor_1,
        rating = 4,
        category = DoctorCategory.PSYCHOLOGY,
        patients = "1.2K",
        experience = "8 yr",
        ratingValue = "4.2",
        description = "Dr. Zainab Saeed is a dedicated clinical psychologist specializing in cognitive behavioral therapy. Her approach combines traditional and modern techniques to treat anxiety, depression, and stress-related disorders."
    ),
    Doctor(
        id = 1,
        name = "Dr. Fatima Murtaza",
        specialty = "Clinical Psychology",
        imageRes = R.drawable.doctor_2,
        rating = 3,
        category = DoctorCategory.PSYCHOLOGY,
        patients = "800+",
        experience = "5 yr",
        ratingValue = "3.8",
        description = "Dr. Fatima Murtaza specializes in family therapy and relationship counseling. She creates a supportive environment where families can work together to overcome challenges and improve communication."
    ),
    Doctor(
        id = 2,
        name = "Dr. Sarah Khan",
        specialty = "Child Psychology",
        imageRes = R.drawable.doctor_5,
        rating = 5,
        category = DoctorCategory.PSYCHOLOGY,
        patients = "1K+",
        experience = "10 yr",
        ratingValue = "4.8",
        description = "Dr. Sarah Khan is an experienced child psychologist specializing in early childhood development and behavioral disorders. She expertly handles ADHD, autism, and learning disabilities in children."
    ),
    Doctor(
        id = 3,
        name = "Dr. Ahmed Khan",
        specialty = "Cardiology",
        imageRes = R.drawable.doctor_3,
        rating = 5,
        category = DoctorCategory.CARDIOLOGY,
        patients = "2.5K",
        experience = "15 yr",
        ratingValue = "4.9",
        description = "Dr. Ahmed Khan is a renowned cardiologist treating complex cardiovascular conditions. He specializes in advanced cardiac imaging and heart disease management with a focus on prevention."
    ),
    Doctor(
        id = 4,
        name = "Dr. Ali Hassan",
        specialty = "Interventional Cardiology",
        imageRes = R.drawable.doctor_6,
        rating = 4,
        category = DoctorCategory.CARDIOLOGY,
        patients = "1.8K",
        experience = "12 yr",
        ratingValue = "4.5",
        description = "Dr. Ali Hassan is a skilled interventional cardiologist specializing in minimally invasive cardiac procedures. His expertise includes coronary angioplasty and stent placement."
    ),
    Doctor(
        id = 5,
        name = "Dr. Sara Ali",
        specialty = "General Dentistry",
        imageRes = R.drawable.doctor_4,
        rating = 4,
        category = DoctorCategory.DENTIST,
        patients = "1.5K",
        experience = "7 yr",
        ratingValue = "4.3",
        description = "Dr. Sara Ali is a gentle general dentist focusing on preventive care and patient education. She specializes in routine procedures, cosmetic dentistry, and treating dental anxiety."
    ),
    Doctor(
        id = 6,
        name = "Dr. Aisha Malik",
        specialty = "Orthodontics",
        imageRes = R.drawable.doctor_7,
        rating = 5,
        category = DoctorCategory.DENTIST,
        patients = "2K+",
        experience = "9 yr",
        ratingValue = "4.7",
        description = "Dr. Aisha Malik is a leading orthodontist specializing in both traditional braces and clear aligners. She expertly handles complex cases using the latest 3D imaging technology."
    )
    // Add more doctors as needed
) 