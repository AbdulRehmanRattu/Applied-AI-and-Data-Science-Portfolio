package com.example.sgpc_app.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.annotation.DrawableRes
import com.example.sgpc_app.R
import com.example.sgpc_app.data.Doctor
import com.example.sgpc_app.components.FooterIcon

@Composable
fun BookingSuccessScreen(
    doctor: Doctor,
    onViewAppointment: () -> Unit,
    onHomeClick: () -> Unit,
    onCalendarClick: () -> Unit,
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
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(48.dp))

            // Success Icon
            Image(
                painter = painterResource(id = R.drawable.ic_success_check),
                contentDescription = "Success",
                modifier = Modifier.size(64.dp)
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Booking Successful Text
            Text(
                text = "Booking",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Successful",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Thank You",
                fontSize = 32.sp,
                color = Color(0xFF4169E1),
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Doctor Card
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Doctor Image
                    Image(
                        painter = painterResource(id = doctor.imageRes),
                        contentDescription = "Doctor Image",
                        modifier = Modifier
                            .size(160.dp)
                            .clip(CircleShape),
                        contentScale = ContentScale.Crop
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    // Doctor Name
                    Text(
                        text = doctor.name,
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Bold
                    )

                    // Doctor Specialty
                    Text(
                        text = doctor.specialty,
                        fontSize = 16.sp,
                        color = Color(0xFF4169E1)
                    )

                    Spacer(modifier = Modifier.height(24.dp))

                    // Communication Icons Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CommunicationIcon(
                            icon = R.drawable.ic_chat,
                            contentDescription = "Message"
                        )
                        Spacer(modifier = Modifier.width(32.dp))
                        CommunicationIcon(
                            icon = R.drawable.ic_call,
                            contentDescription = "Call"
                        )
                        Spacer(modifier = Modifier.width(32.dp))
                        CommunicationIcon(
                            icon = R.drawable.ic_video,
                            contentDescription = "Video"
                        )
                    }

                    Spacer(modifier = Modifier.height(24.dp))
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
                Box(modifier = Modifier.clickable { onCalendarClick() }) {
                    FooterIcon(icon = R.drawable.ic_calendar, isSelected = false)
                }
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
private fun CommunicationIcon(
    @DrawableRes icon: Int,
    contentDescription: String
) {
    Box(
        modifier = Modifier
            .size(40.dp)
            .clip(RoundedCornerShape(8.dp))
            .background(Color.White)
            .padding(8.dp),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            painter = painterResource(id = icon),
            contentDescription = contentDescription,
            tint = Color.Black,
            modifier = Modifier.size(24.dp)
        )
    }
} 