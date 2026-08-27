package com.example.sgpc_app.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sgpc_app.R
import com.example.sgpc_app.components.FooterIcon

@Composable
fun SettingsScreen(
    onHomeClick: () -> Unit,
    onCalendarClick: () -> Unit,
    onProfileClick: () -> Unit
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
                text = "Settings",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 16.dp)
            )

            // Settings Options
            SettingOption(
                icon = R.drawable.ic_notifications,
                title = "Notifications",
                hasSwitch = true
            )
            SettingOption(
                icon = R.drawable.ic_lock,
                title = "Privacy",
                hasSwitch = false
            )
            SettingOption(
                icon = R.drawable.ic_help,
                title = "Help Center",
                hasSwitch = false
            )
            SettingOption(
                icon = R.drawable.ic_info,
                title = "About Us",
                hasSwitch = false
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
                Box(modifier = Modifier.clickable { onProfileClick() }) {
                    FooterIcon(icon = R.drawable.ic_profile, isSelected = false)
                }
                FooterIcon(icon = R.drawable.ic_settings, isSelected = true)
            }
        }
    }
}

@Composable
private fun SettingOption(
    icon: Int,
    title: String,
    hasSwitch: Boolean
) {
    var switchState by remember { mutableStateOf(false) }

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
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
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
            if (hasSwitch) {
                Switch(
                    checked = switchState,
                    onCheckedChange = { switchState = it }
                )
            }
        }
    }
} 