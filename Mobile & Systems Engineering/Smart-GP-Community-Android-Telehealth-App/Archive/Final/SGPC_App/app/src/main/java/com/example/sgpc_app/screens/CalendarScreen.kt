package com.example.sgpc_app.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sgpc_app.R
import com.example.sgpc_app.components.FooterIcon
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@Composable
fun CalendarScreen(
    onHomeClick: () -> Unit,
    onProfileClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 16.dp, end = 16.dp, top = 56.dp)
        ) {
            Text(
                text = "My Appointments",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 16.dp)
            )

            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(sampleAppointments) { appointment ->
                    AppointmentCard(appointment)
                }
            }
        }

        // Footer
        Surface(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .height(60.dp),
            color = Color(0xFFB5E1F3)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 32.dp),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(modifier = Modifier.clickable { onHomeClick() }) {
                    FooterIcon(icon = R.drawable.ic_home, isSelected = false)
                }
                FooterIcon(icon = R.drawable.ic_calendar, isSelected = true)
                Box(modifier = Modifier.clickable { onProfileClick() }) {
                    FooterIcon(icon = R.drawable.ic_profile, isSelected = false)
                }
                Box(modifier = Modifier.clickable { onSettingsClick() }) {
                    FooterIcon(icon = R.drawable.ic_settings, isSelected = false)
                }
            }
        }
    }
}

@Composable
private fun AppointmentCard(appointment: Appointment) {
    Card(
        modifier = Modifier
            .fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = appointment.doctorName,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = appointment.specialty,
                fontSize = 14.sp,
                color = Color.Gray
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Date: ${appointment.dateTime.format(DateTimeFormatter.ofPattern("MMM dd, yyyy"))}",
                fontSize = 14.sp
            )
            Text(
                text = "Time: ${appointment.dateTime.format(DateTimeFormatter.ofPattern("hh:mm a"))}",
                fontSize = 14.sp
            )
            Text(
                text = "Status: ${appointment.status}",
                fontSize = 14.sp,
                color = when(appointment.status) {
                    "Upcoming" -> Color(0xFF4CAF50)
                    "Completed" -> Color.Gray
                    else -> Color.Red
                }
            )
        }
    }
}

data class Appointment(
    val doctorName: String,
    val specialty: String,
    val dateTime: LocalDateTime,
    val status: String
)

private val sampleAppointments = listOf(
    Appointment(
        doctorName = "Dr. Zainab Saeed",
        specialty = "Clinical Psychology",
        dateTime = LocalDateTime.now().plusDays(2),
        status = "Upcoming"
    ),
    Appointment(
        doctorName = "Dr. Ahmed Khan",
        specialty = "Cardiology",
        dateTime = LocalDateTime.now().minusDays(1),
        status = "Completed"
    ),
    // Add more sample appointments
) 