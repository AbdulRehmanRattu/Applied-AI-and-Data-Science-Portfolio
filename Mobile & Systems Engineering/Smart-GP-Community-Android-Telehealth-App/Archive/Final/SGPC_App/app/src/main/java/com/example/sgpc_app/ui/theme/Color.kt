package com.example.sgpc_app.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

val Primary = Color(0xFF2196F3)
val PrimaryDark = Color(0xFF1976D2)
val Secondary = Color(0xFF03DAC5)
val Background = Color(0xFFFFFFFF)

// Light Theme Colors
val LightColorScheme = lightColorScheme(
    primary = Primary,
    secondary = Secondary,
    background = Background
)

// Dark Theme Colors
val DarkColorScheme = darkColorScheme(
    primary = PrimaryDark,
    secondary = Secondary,
    background = Color.Black
)