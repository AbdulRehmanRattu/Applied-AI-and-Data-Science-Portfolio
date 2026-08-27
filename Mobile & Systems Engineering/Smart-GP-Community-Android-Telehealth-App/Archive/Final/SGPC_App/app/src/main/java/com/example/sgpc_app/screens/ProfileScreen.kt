package com.example.sgpc_app.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sgpc_app.R
import com.example.sgpc_app.components.FooterIcon

@Composable
fun ProfileScreen(
    onHomeClick: () -> Unit,
    onCalendarClick: () -> Unit,
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
                .padding(start = 16.dp, end = 16.dp, top = 56.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Profile Image
            Image(
                painter = painterResource(id = R.drawable.profile_pic),
                contentDescription = "Profile Picture",
                modifier = Modifier
                    .size(120.dp)
                    .clip(CircleShape),
                contentScale = ContentScale.Crop
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "John Doe",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "john.doe@example.com",
                fontSize = 16.sp,
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Profile Options
            ProfileOption(
                icon = R.drawable.ic_profile,
                title = "Personal Information"
            )
            ProfileOption(
                icon = R.drawable.ic_calendar,
                title = "My Appointments"
            )
            ProfileOption(
                icon = R.drawable.ic_settings,
                title = "Settings"
            )
            ProfileOption(
                icon = R.drawable.ic_help,
                title = "Help & Support"
            )
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
                FooterIcon(icon = R.drawable.ic_profile, isSelected = true)
                Box(modifier = Modifier.clickable { onSettingsClick() }) {
                    FooterIcon(icon = R.drawable.ic_settings, isSelected = false)
                }
            }
        }
    }
}

@Composable
private fun ProfileOption(
    icon: Int,
    title: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                painter = painterResource(id = icon),
                contentDescription = title,
                tint = Color(0xFF4169E1)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Text(text = title)
        }
    }
} 