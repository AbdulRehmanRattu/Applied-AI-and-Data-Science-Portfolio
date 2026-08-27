package com.example.sgpc_app.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
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
import androidx.annotation.DrawableRes
import com.example.sgpc_app.R
import com.example.sgpc_app.data.Doctor
import com.example.sgpc_app.data.DoctorCategory
import com.example.sgpc_app.data.getDoctorsByCategory
import androidx.compose.material3.TextFieldDefaults
import com.example.sgpc_app.data.allDoctors
import com.example.sgpc_app.components.FooterIcon
import androidx.navigation.NavController

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onDoctorClick: (Int) -> Unit,
    navController: NavController,
    onCalendarClick: () -> Unit,
    onProfileClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    var searchQuery by remember { mutableStateOf("") }
    var selectedCategory by remember { mutableStateOf(DoctorCategory.PSYCHOLOGY) }

    // Add filtered doctors state
    val filteredDoctors by remember(searchQuery, selectedCategory) {
        derivedStateOf {
            val doctorsInCategory = getDoctorsByCategory(selectedCategory)
            if (searchQuery.isEmpty()) {
                doctorsInCategory
            } else {
                doctorsInCategory.filter { doctor ->
                    doctor.name.contains(searchQuery, ignoreCase = true) ||
                    doctor.specialty.contains(searchQuery, ignoreCase = true)
                }
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 56.dp, start = 16.dp, end = 16.dp, bottom = 80.dp)
        ) {
            // Top Bar with Menu and Profile
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = { }) {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_menu),
                        contentDescription = "Menu"
                    )
                }
                
                Text(
                    text = "Hello, User",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
                
                Image(
                    painter = painterResource(id = R.drawable.profile_pic),
                    contentDescription = "Profile",
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape),
                    contentScale = ContentScale.Crop
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Updated Search Bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Search Doctor") },
                leadingIcon = {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_search),
                        contentDescription = "Search"
                    )
                },
                trailingIcon = {
                    if (searchQuery.isNotEmpty()) {
                        IconButton(onClick = { searchQuery = "" }) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_clear),
                                contentDescription = "Clear search"
                            )
                        }
                    } else {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_filter),
                            contentDescription = "Filter"
                        )
                    }
                },
                shape = RoundedCornerShape(8.dp),
                colors = TextFieldDefaults.outlinedTextFieldColors(
                    containerColor = Color.LightGray.copy(alpha = 0.2f)
                ),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Categories Title with result count
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Doctors",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
                if (searchQuery.isNotEmpty()) {
                    Text(
                        text = "${filteredDoctors.size} found",
                        color = Color.Gray,
                        fontSize = 14.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Categories Slider
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(DoctorCategory.values()) { category ->
                    CategoryChip(
                        category = category,
                        isSelected = category == selectedCategory,
                        onSelect = { 
                            selectedCategory = category
                            // Optionally clear search when changing category
                            // searchQuery = ""
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (filteredDoctors.isEmpty()) {
                // No results view
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_search),
                            contentDescription = null,
                            modifier = Modifier.size(48.dp),
                            tint = Color.Gray
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "No doctors found",
                            color = Color.Gray,
                            fontSize = 16.sp
                        )
                        Text(
                            text = "Try adjusting your search",
                            color = Color.Gray.copy(alpha = 0.7f),
                            fontSize = 14.sp
                        )
                    }
                }
            } else {
                // Doctors Grid with filtered results
                LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    horizontalArrangement = Arrangement.spacedBy(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    items(filteredDoctors) { doctor ->
                        DoctorCard(
                            doctor = doctor,
                            onDoctorClick = { onDoctorClick(allDoctors.indexOf(doctor)) }
                        )
                    }
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
                FooterIcon(
                    icon = R.drawable.ic_home,
                    isSelected = true
                )
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

@Composable
fun CategoryChip(
    category: DoctorCategory,
    isSelected: Boolean,
    onSelect: () -> Unit
) {
    Surface(
        modifier = Modifier.clickable(onClick = onSelect),
        shape = RoundedCornerShape(20.dp),
        color = if (isSelected) Color(0xFF4CAF50) else Color.LightGray.copy(alpha = 0.2f)
    ) {
        Text(
            text = category.displayName,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            color = if (isSelected) Color.White else Color.Black
        )
    }
}

@Composable
fun DoctorCard(
    doctor: Doctor,
    onDoctorClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(0.8f)
            .clickable(onClick = onDoctorClick),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column {
            Image(
                painter = painterResource(id = doctor.imageRes),
                contentDescription = "Doctor Image",
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentScale = ContentScale.Crop
            )
            
            Column(
                modifier = Modifier.padding(8.dp)
            ) {
                Text(
                    text = doctor.name,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
                
                Text(
                    text = doctor.specialty,
                    color = Color.Gray,
                    fontSize = 14.sp
                )
                
                Row(
                    modifier = Modifier.padding(vertical = 4.dp)
                ) {
                    repeat(doctor.rating) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_star_filled),
                            contentDescription = "Star",
                            tint = Color(0xFFFFD700),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                    repeat(5 - doctor.rating) {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_star_outline),
                            contentDescription = "Empty Star",
                            tint = Color.Gray,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }
    }
} 