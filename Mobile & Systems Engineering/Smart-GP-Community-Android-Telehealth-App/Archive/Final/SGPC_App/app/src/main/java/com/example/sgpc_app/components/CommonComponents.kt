package com.example.sgpc_app.components

import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.annotation.DrawableRes

@Composable
fun FooterIcon(
    @DrawableRes icon: Int,
    isSelected: Boolean
) {
    Icon(
        painter = painterResource(id = icon),
        contentDescription = null,
        modifier = Modifier.size(24.dp),
        tint = if (isSelected) Color.Black else Color.Gray
    )
} 