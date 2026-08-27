package com.example.sgpc_app.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.example.sgpc_app.screens.*
import com.example.sgpc_app.data.allDoctors
import androidx.compose.foundation.clickable

@Composable
fun NavGraph(navController: NavHostController) {
    NavHost(
        navController = navController,
        startDestination = "splash"
    ) {
        composable("splash") {
            SplashScreen()
        }
        composable("welcome") {
            WelcomeScreen(
                onGetStartedClick = {
                    navController.navigate("signin") {
                        popUpTo("splash") { inclusive = true }
                    }
                }
            )
        }
        composable("signin") {
            SignInScreen(
                onCreateAccountClick = {
                    navController.navigate("signup")
                },
                onSignInClick = {
                    navController.navigate("login")
                }
            )
        }
        composable("signup") {
            SignUpScreen(
                onSignUpSuccess = {
                    navController.navigate("home") {
                        popUpTo("signup") { inclusive = true }
                    }
                },
                onLoginClick = {
                    navController.navigate("login") {
                        popUpTo("signup") { inclusive = true }
                    }
                }
            )
        }
        composable("login") {
            LoginScreen(
                onLoginSuccess = {
                    navController.navigate("home") {
                        popUpTo("login") { inclusive = true }
                    }
                },
                onSignUpClick = {
                    navController.navigate("signup") {
                        popUpTo("login") { inclusive = true }
                    }
                }
            )
        }
        composable("home") {
            HomeScreen(
                onDoctorClick = { doctorId ->
                    navController.navigate("doctorProfile/$doctorId")
                },
                navController = navController,
                onCalendarClick = { navController.navigate("calendar") },
                onProfileClick = { navController.navigate("profile") },
                onSettingsClick = { navController.navigate("settings") }
            )
        }
        composable(
            route = "doctorProfile/{doctorId}",
            arguments = listOf(navArgument("doctorId") { type = NavType.IntType })
        ) { backStackEntry ->
            val doctorId = backStackEntry.arguments?.getInt("doctorId") ?: 0
            val doctor = allDoctors[doctorId]
            DoctorProfileScreen(
                doctor = doctor,
                onBackClick = { navController.navigateUp() },
                onConsultClick = { navController.navigate("bookAppointment/$doctorId") },
                onHomeClick = { navController.navigate("home") },
                onCalendarClick = { navController.navigate("calendar") },
                onProfileClick = { navController.navigate("profile") },
                onSettingsClick = { navController.navigate("settings") }
            )
        }
        composable(
            route = "bookAppointment/{doctorId}",
            arguments = listOf(navArgument("doctorId") { type = NavType.IntType })
        ) { backStackEntry ->
            val doctorId = backStackEntry.arguments?.getInt("doctorId") ?: 0
            val doctor = allDoctors[doctorId]
            BookAppointmentScreen(
                doctor = doctor,
                onBackClick = { 
                    navController.popBackStack()
                },
                onBookAppointment = {
                    navController.navigate("payment/$doctorId")
                }
            )
        }
        composable(
            route = "bookingSuccess/{doctorId}",
            arguments = listOf(navArgument("doctorId") { type = NavType.IntType })
        ) { backStackEntry ->
            val doctorId = backStackEntry.arguments?.getInt("doctorId") ?: 0
            val doctor = allDoctors[doctorId]
            BookingSuccessScreen(
                doctor = doctor,
                onViewAppointment = { navController.navigate("calendar") },
                onHomeClick = { navController.navigate("home") },
                onCalendarClick = { navController.navigate("calendar") },
                onProfileClick = { navController.navigate("profile") },
                onSettingsClick = { navController.navigate("settings") }
            )
        }
        composable(
            route = "payment/{doctorId}",
            arguments = listOf(navArgument("doctorId") { type = NavType.IntType })
        ) { backStackEntry ->
            val doctorId = backStackEntry.arguments?.getInt("doctorId") ?: 0
            PaymentScreen(
                doctorId = doctorId,
                onBackClick = { navController.popBackStack() },
                onPaymentComplete = { id ->
                    navController.navigate("bookingSuccess/$id") {
                        popUpTo("home")
                    }
                }
            )
        }
        composable("calendar") {
            CalendarScreen(
                onHomeClick = { navController.navigate("home") },
                onProfileClick = { navController.navigate("profile") },
                onSettingsClick = { navController.navigate("settings") }
            )
        }
        composable("profile") {
            ProfileScreen(
                onHomeClick = { navController.navigate("home") },
                onCalendarClick = { navController.navigate("calendar") },
                onSettingsClick = { navController.navigate("settings") }
            )
        }
        composable("settings") {
            SettingsScreen(
                onHomeClick = { navController.navigate("home") },
                onCalendarClick = { navController.navigate("calendar") },
                onProfileClick = { navController.navigate("profile") }
            )
        }
    }
} 