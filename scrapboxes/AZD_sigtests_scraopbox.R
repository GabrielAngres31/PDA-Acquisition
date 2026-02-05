# Required package for Dunn's test:
# If you don't have it installed, uncomment and run the line below:
# install.packages("dunn.test")
library(dunn.test) # Load the package
library(dplyr)

# --- END: Sample Data ---


# --- Main Script Logic ---

my_data <- read.csv("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/clumps/AZD/full.csv")

my_data %<>% mutate(area_u = area*(0.5681818**2), 
                    axis_minor_length_u = axis_minor_length*0.5681818,
                    axis_minor_length_u = axis_minor_length*0.5681818,
                    perimeter_u = perimeter*0.5681818
                    )

# treatment_order <- c("DMSO", "100nM", "250nM", "1uM")
# my_data_raw$treatment <- factor(my_data_raw$treatment, levels = treatment_order, ordered = TRUE)
treat_order <- c("DMSO", "100nM", "250nM", "1uM")
my_data$treatment <- factor(my_data$treatment, levels = treat_order, ordered = TRUE)

cat("--- Kruskal-Wallis Test and Post-Hoc Analysis ---\n\n")

# 1. Perform Kruskal-Wallis Test
# This non-parametric test checks if there's a significant difference
# in the medians (or distributions) of the 'area' variable across the 'ID' groups.
kruskal_result <- kruskal.test(area ~ treatment, data = my_data)

cat("Kruskal-Wallis Rank Sum Test Results:\n")
cat(paste0("  Statistic (Chi-squared): ", round(kruskal_result$statistic, 4), "\n"))
cat(paste0("  Degrees of Freedom: ", kruskal_result$parameter, "\n"))
cat(paste0("  P-value: ", format.pval(kruskal_result$p.value, digits = 4), "\n"))

# Interpretation of Kruskal-Wallis
if (kruskal_result$p.value < 0.05) {
  cat("\nInterpretation: There is a statistically significant difference in 'area' across treatment groups (p < 0.05).\n")
  cat("Proceeding with post-hoc analysis (Dunn's Test) to find specific group differences.\n\n")
  
  dunn_result <- dunn.test(
    x = my_data$axis_minor_length_u,             #ALTER
    g = my_data$treatment,
    method = "bonferroni", # Common adjustment for multiple comparisons
    altp = TRUE             # Display adjusted p-values
  )
  
  cat("Dunn's Test for Multiple Comparisons (Bonferroni p-value adjustment):\n")

  dunn_table <- data.frame(
    Comparison = dunn_result$comparisons,
    Z_Value = round(dunn_result$Z, 3),
    P_Value = format.pval(dunn_result$altP, digits = 4),
    Adjusted_P_Value = format.pval(dunn_result$altP.adjusted, digits = 4) # P is the adjusted p-value
  )
  
  # Add a significance column
  dunn_table$Significant <- ifelse(dunn_result$altP < 0.05, "!", "--")
  dunn_table$AltSig <- ifelse(dunn_result$altP.adjusted < 0.05, "!", "--")
  
  
  dunn_table_small <- subset(dunn_table, Adjusted_P_Value < 0.10)
  # Print the organized table
  print(dunn_table, row.names = FALSE) # Suppress row names for cleaner output
  cat("\n")
  
  cat("Interpretation of Dunn's Test:\n")
  cat("  'Significant' column indicates if the adjusted p-value for the comparison is < 0.05.\n")
  cat("  A 'YES' indicates a significant difference between the two compared groups.\n")
  
} else {
  cat("\nInterpretation: There is NO statistically significant difference in 'area' across ID groups (p >= 0.05).\n")
  cat("Post-hoc analysis (Dunn's Test) is not necessary.\n")
}

cat("\n--- Testing Means ---\n")


grouped_data_for_KW <- my_data %>%
  group_by(ID, treatment) %>%
  summarise(
    # mean_response = mean(area_u, na.rm = TRUE),
    mean_stomatal_axis_minor_length_u = mean(axis_minor_length_u, na.rm = TRUE), #ALTER
    # spec_area = mean_leaf_area/mean_response,
    n_individuals_in_group = n(),
    .groups = 'drop'
  ) %>%
  ungroup()


kruskal_result_mean <- kruskal.test(mean_stomatal_axis_minor_length_u ~ treatment, data = grouped_data_for_KW) #ALTER

cat("Kruskal-Wallis Rank Sum Test Results:\n")
cat(paste0("  Statistic (Chi-squared): ", round(kruskal_result_mean$statistic, 4), "\n"))
cat(paste0("  Degrees of Freedom: ", kruskal_result_mean$parameter, "\n"))
cat(paste0("  P-value: ", format.pval(kruskal_result_mean$p.value, digits = 4), "\n"))

# Interpretation of Kruskal-Wallis
if (kruskal_result_mean$p.value < 0.05) {
  cat("\nInterpretation: There is a statistically significant difference in 'area' across treatment groups (p < 0.05).\n")
  cat("Proceeding with post-hoc analysis (Dunn's Test) to find specific group differences.\n\n")
  
  print(length(grouped_data_for_KW$mean_stomatal_axis_minor_length_u)) #ALTER
  print(length(grouped_data_for_KW$treatment))
  
  dunn_result_mean <- dunn.test(
    x = grouped_data_for_KW$mean_stomatal_axis_minor_length_u, #ALTER
    g = grouped_data_for_KW$treatment,
    method = "bonferroni", # Common adjustment for multiple comparisons
    altp = TRUE             # Display adjusted p-values
  )
  
  cat("Dunn's Test for Multiple Comparisons (Bonferroni p-value adjustment):\n")
  
  dunn_table_mean <- data.frame(
    Comparison = dunn_result_mean$comparisons,
    Z_Value = round(dunn_result_mean$Z, 3),
    P_Value = format.pval(dunn_result_mean$altP, digits = 4),
    Adjusted_P_Value = format.pval(dunn_result_mean$altP.adjusted, digits = 4) # P is the adjusted p-value
  )
  
  # Add a significance column
  dunn_table_mean$Significant <- ifelse(dunn_result_mean$altP < 0.05, "!", "--")
  dunn_table_mean$AltSig <- ifelse(dunn_result_mean$altP.adjusted < 0.05, "!", "--")
  
  # Print the organized table
  print(dunn_table_mean, row.names = FALSE) # Suppress row names for cleaner output
  cat("\n")
  
  cat("Interpretation of Dunn's Test:\n")
  cat("  'Significant' column indicates if the adjusted p-value for the comparison is < 0.05.\n")
  cat("  A 'YES' indicates a significant difference between the two compared groups.\n")
  
} else {
  cat("\nInterpretation: There is NO statistically significant difference in 'area' means across ID groups (p >= 0.05).\n")
  cat("Post-hoc analysis (Dunn's Test) is not necessary.\n")
}

cat("\n--- End of Analysis ---\n")