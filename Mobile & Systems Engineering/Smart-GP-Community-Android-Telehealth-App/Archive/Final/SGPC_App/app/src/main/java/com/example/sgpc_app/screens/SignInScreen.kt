package com.example.sgpc_app.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sgpc_app.R
import com.example.sgpc_app.ui.theme.Primary

@Composable
fun SignInScreen(
    onCreateAccountClick: () -> Unit,
    onSignInClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Top logo
        Image(
            painter = painterResource(id = R.drawable.stethoscope),
            contentDescription = "App Logo",
            modifier = Modifier
                .size(70.dp)
                .padding(top = 26.dp) // Adjusted top padding to bring the logo down
        )

        Spacer(modifier = Modifier.height(32.dp)) // Reduced height between logo and text

        // Header section with text and image
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Join Us for a Healthier Tomorrow. Your Information is Secure with Us.",
                fontSize = 24.sp,
                lineHeight = 32.sp,
                modifier = Modifier.weight(1f)
            )

            Image(
                painter = painterResource(id = R.drawable.lady_logo),
                contentDescription = "Healthcare Professional",
                modifier = Modifier.size(121.dp, 159.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp)) // Reduced height between header and community health logo

        // Community Health Logo
        Image(
            painter = painterResource(id = R.drawable.community_health_logo),
            contentDescription = "Community Health Logo",
            modifier = Modifier.size(200.dp)
        )

        Spacer(modifier = Modifier.height(24.dp)) // Reduced height between community logo and terms text

        // Terms text
        Text(
            buildAnnotatedString {
                append("By tapping 'Sign in' you agree to our ")
                withStyle(SpanStyle(color = Primary)) {
                    append("Terms")
                }
                append(".\nLearn how we process your data in our ")
                withStyle(SpanStyle(color = Primary)) {
                    append("Privacy Policy")
                }
                append(" and ")
                withStyle(SpanStyle(color = Primary)) {
                    append("Cookies Policy")
                }
                append(".")
            },
            textAlign = TextAlign.Center,
            fontSize = 12.sp,
            lineHeight = 16.sp
        )

        Spacer(modifier = Modifier.height(16.dp)) // Reduced height between terms text and buttons

        // Create Account Button
        Button(
            onClick = onCreateAccountClick,
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            shape = RoundedCornerShape(24.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Primary
            )
        ) {
            Text(
                "CREATE ACCOUNT",
                fontSize = 16.sp
            )
        }

        Spacer(modifier = Modifier.height(8.dp)) // Reduced height between buttons

        // Sign In Button
        OutlinedButton(
            onClick = onSignInClick,
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            shape = RoundedCornerShape(24.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = Primary
            )
        ) {
            Text(
                "SIGN IN",
                fontSize = 16.sp
            )
        }

        Spacer(modifier = Modifier.height(32.dp))
    }
}