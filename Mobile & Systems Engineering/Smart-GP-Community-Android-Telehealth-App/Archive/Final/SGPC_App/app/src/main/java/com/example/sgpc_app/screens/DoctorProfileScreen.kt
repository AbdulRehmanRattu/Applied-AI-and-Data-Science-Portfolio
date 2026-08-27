package com.example.sgpc_app.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.material3.MaterialTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DoctorProfileScreen(
    doctor: Doctor,
    onBackClick: () -> Unit,
    onConsultClick: () -> Unit,
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
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
        ) {
            // Add top margin
            Spacer(modifier = Modifier.height(24.dp))
            
            // Doctor Image Section
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
            ) {
                Image(
                    painter = painterResource(id = doctor.imageRes),
                    contentDescription = "Doctor Image",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                
                // Back Button with clickable modifier
                Box(
                    modifier = Modifier
                        .padding(16.dp)
                        .size(36.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color.White.copy(alpha = 0.7f))
                        .clickable(onClick = onBackClick)  // Make sure click is handled
                        .padding(8.dp)
                ) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_back),
                        contentDescription = "Back",
                        modifier = Modifier.size(20.dp),
                        tint = Color.Black
                    )
                }
            }

            // Doctor Info Section
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp)
            ) {
                Text(
                    text = doctor.name,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                
                Text(
                    text = doctor.specialty,
                    fontSize = 14.sp,
                    color = Color.Blue,
                    modifier = Modifier.padding(vertical = 4.dp)
                )

                // Communication Options
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 12.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CommunicationOption(
                        icon = R.drawable.ic_chat,
                        label = "Message",
                        modifier = Modifier.weight(1f)
                    )
                    CommunicationOption(
                        icon = R.drawable.ic_call,
                        label = "Call",
                        modifier = Modifier.weight(1f)
                    )
                    CommunicationOption(
                        icon = R.drawable.ic_video,
                        label = "Video",
                        modifier = Modifier.weight(1f)
                    )
                }

                // Stats Section
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    StatItem("Patients", doctor.patients)
                    StatItem("Experience", doctor.experience)
                    StatItem("Rating", doctor.ratingValue)
                }

                // Description
                Text(
                    text = doctor.description,
                    color = Color.Gray,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    textAlign = TextAlign.Justify,
                    style = MaterialTheme.typography.bodyMedium.copy(
                        textAlign = TextAlign.Justify
                    )
                )
            }
        }

        // Bottom Section with Consultation Button and Footer
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
        ) {
            // Consultation Button
            Button(
                onClick = onConsultClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF4CAF50)
                )
            ) {
                Text(
                    text = "Get Consultation",
                    modifier = Modifier.padding(vertical = 4.dp),
                    fontSize = 16.sp
                )
            }

            // Footer
            Surface(
                modifier = Modifier
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
                        FooterIcon(
                            icon = R.drawable.ic_home,
                            isSelected = false
                        )
                    }
                    Box(modifier = Modifier.clickable { onCalendarClick() }) {
                        FooterIcon(
                            icon = R.drawable.ic_calendar,
                            isSelected = false
                        )
                    }
                    Box(modifier = Modifier.clickable { onProfileClick() }) {
                        FooterIcon(
                            icon = R.drawable.ic_profile,
                            isSelected = false
                        )
                    }
                    Box(modifier = Modifier.clickable { onSettingsClick() }) {
                        FooterIcon(
                            icon = R.drawable.ic_settings,
                            isSelected = false
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun CommunicationOption(
    icon: Int,
    label: String,
    modifier: Modifier = Modifier
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
    ) {
        Icon(
            painter = painterResource(id = icon),
            contentDescription = label,
            modifier = Modifier.size(20.dp),  // Made icon smaller
            tint = Color.Black
        )
        Text(
            text = label,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 4.dp)
        )
    }
}

@Composable
private fun StatItem(
    label: String,
    value: String
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            fontSize = 16.sp,  // Made text smaller
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            fontSize = 12.sp,  // Made text smaller
            color = Color.Gray
        )
    }
}