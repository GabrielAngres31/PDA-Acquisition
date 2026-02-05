# Title: R Script for Box Plot Matrix with Kruskal-Wallis and Dunn's Post-Hoc Test

# Description:
# This script loads data from a specified CSV file into a dataframe.
# It reshapes the data to a 'long' format to facilitate plotting multiple measures.
# It then generates a matrix of box and whisker plots.
# Each row in the plot matrix corresponds to one of the specified value columns (measures).
# Within each plot, the x-axis represents the grouping column (treatments).
# Significance is evaluated through a Kruskal-Wallis test for overall differences
# between treatments for each measure. If the Kruskal-Wallis test is significant,
# Dunn's post-hoc test is performed for pairwise comparisons between treatments
# for that specific measure.
# Significance annotations (e.g., *, **, ***) are added to the plots.

# Required Packages:
# ggplot2: For creating high-quality statistical graphics.
# ggpubr: For easily adding p-values and significance bars to ggplot2 plots.
# rstatix: Provides pipe-friendly R functions for easy statistical analyses,
#          including Kruskal-Wallis and Dunn's test.
# tidyr: For reshaping data from wide to long format using pivot_longer.

# --- 1. Install and Load Required Packages ---
# Check if ggplot2 is installed, if not, install it
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
# Check if ggpubr is installed, if not, install it
if (!requireNamespace("ggpubr", quietly = TRUE)) {
  install.packages("ggpubr")
}
# Check if rstatix is installed, if not, install it
if (!requireNamespace("rstatix", quietly = TRUE)) {
  install.packages("rstatix")
}
# Check if tidyr is installed, if not, install it
if (!requireNamespace("tidyr", quietly = TRUE)) {
  install.packages("tidyr")
}

# Load the packages
library(ggplot2)
library(ggpubr)
library(rstatix) # For kruskal_test and dunn_test
library(tidyr)   # For pivot_longer

# --- 2. Define File Path and Column Names ---
# IMPORTANT: Replace "your_data.csv" with the actual path to your file.
# Example: filepath <- "C:/Users/YourUser/Documents/my_data.csv"



#-------------------------------------------


# # filepath <- "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/publication_data.csv" # <--- EDIT THIS LINE WITH YOUR FILE PATH
# filepath <- "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/areas_WT_EXTENSION.csv" # <--- EDIT THIS LINE WITH YOUR FILE PATH
# 
# # filepath <- "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/clumps/AZD/full.csv" # <--- EDIT THIS LINE WITH YOUR FILE PATH
# 
# # Define the column name for grouping variable
# 
# # group_column_name <- "treatment" # <--- EDIT THIS LINE if your grouping column has a different name
# group_column_name <- "age" # <--- EDIT THIS LINE if your grouping column has a different name
# 
# # IMPORTANT: Define the names of the value columns you want to plot.
# # These will become the 'rows' in your plot matrix.
# 
# # value_column_names <- c("corr_stoma_count", "leaf_area_u_sq", "specific_area") # <--- EDIT THIS LINE with your desired value columns
# value_column_names <- c("stoma_count", "leaf_area_u_sq", "density") # <--- EDIT THIS LINE with your desired value columns
# 
# # value_column_names <- c("area", "perimeter", "axis_major_length", "axis_minor_length") # <--- EDIT THIS LINE with your desired value columns
# # value_column_names <- c("area") # <--- EDIT THIS LINE with your desired value columns
# 
# # IMPORTANT: Define your desired order of levels for the grouping column.
# # This ensures a consistent order on the x-axis of your plots.
# 
# # desired_order_of_groups <- c("DMSO", "100nM", "250nM", "1uM") # <--- EDIT THIS LINE with your desired order
# desired_order_of_groups <- c("3", "4", "5", "7") # <--- EDIT THIS LINE with your desired order

# --- 3. Load Data into a Dataframe ---
# This example assumes a CSV file. Adjust `read.csv()` parameters as needed.

# analyzer(filepath, desired_order_of_groups, group_column_name, value_column_names)

analyzer <- function(df_path, ordering, groupcol, valcols) {
  
  tryCatch({
    data_df <- read.csv(df_path, header = TRUE) # header=TRUE assumes first row is column names
    cat("Successfully loaded data from:", df_path, "\n")
    cat("Dimensions of data:", dim(data_df)[1], "rows,", dim(data_df)[2], "columns\n")
  }, error = function(e) {
    stop(paste("Error loading data:", e$message,
               "\nPlease ensure the file path is correct and the file format matches read.csv()."))
  })
  
  trimws(colnames(data_df))
  
  # --- 4. Prepare Data for Plotting and Analysis ---
  # Ensure the grouping column exists in the dataframe
  if (!group_column_name %in% colnames(data_df)) {
    stop(paste("Error: Grouping column '", groupcol, "' not found in data.",
               "Please check the column name or your data file.", sep = ""))
  }
  
  # Ensure all specified value columns exist in the dataframe
  missing_value_columns <- setdiff(valcols, colnames(data_df))
  if (length(missing_value_columns) > 0) {
    stop(paste("Error: The following value column(s) were not found in data:",
               paste(missing_value_columns, collapse = ", "),
               "\nPlease check the column names or your data file.", sep = ""))
  }
  
  # Convert the grouping column to a factor with specific level order
  data_df[[groupcol]] <- factor(data_df[[groupcol]], levels = ordering)
  
  # Display first few rows of the prepared data (for verification)
  cat("\nFirst 6 rows of the prepared data:\n")
  print(head(data_df))
  cat("\nLevels of the grouping column in specified order:\n")
  print(levels(data_df[[groupcol]]))
  
  # Reshape data from wide to long format
  long_data_df <- data_df %>%
    pivot_longer(
      cols = all_of(valcols),  # Columns to pivot
      names_to = "Measure",               # New column for original column names
      values_to = "Value"                 # New column for values
    )
  
  # Convert 'Measure' to a factor with a specific order for consistent plotting rows
  long_data_df$Measure <- factor(long_data_df$Measure, levels = valcols)
  
  # cat("\nFirst 6 rows of the long-format data:\n")
  # print(head(long_data_df))
  
  # --- 5. Perform Statistical Tests (Kruskal-Wallis and Dunn's Post-Hoc) ---
  # Initialize empty dataframes to store results for plotting
  pwc_results_for_plot <- data.frame()
  overall_kruskal_p_values <- data.frame()
  
  # Loop through each unique measure to perform tests
  for (current_measure in levels(long_data_df$Measure)) {
    cat(paste("\n--- Analyzing Measure:", current_measure, "---\n"))
    
    # Subset data for the current measure
    subset_data <- long_data_df %>% filter(Measure == current_measure)
    
    # Perform Kruskal-Wallis test
    kruskal_res <- subset_data %>%
      kruskal_test(as.formula(paste("Value ~", group_column_name)))
    
    cat("Kruskal-Wallis Test Result:\n")
    print(kruskal_res)
    
    # Store overall Kruskal-Wallis p-value for plot annotation
    overall_kruskal_p_values <- rbind(
      overall_kruskal_p_values,
      data.frame(
        Measure = current_measure,
        p.value = kruskal_res$p,
        label = paste0("Kruskal-Wallis p = ", format.pval(kruskal_res$p, digits = 3)),
        # Adjusted position: Use relative coordinates or fixed x for left side, top y
        # For example, x = -Inf, y = Inf, hjust = -0.05, vjust = 1.05 for top-left
        # Or, a specific x position in data coordinates if scales are fixed
        # For now, let's try a different relative corner for robustness across scales: top-left.
        x = -Inf, y = Inf, # Changed to top-left corner
        hjust = -0.05, vjust = 1.05 # Adjusted to give some padding
      )
    )
    
    # Perform Dunn's post-hoc test if overall Kruskal-Wallis is significant
    if (kruskal_res$p < 0.05) {
      cat("Significant overall difference detected in ")
      cat(current_measure)
      cat(". Performing Dunn's post-hoc test.\n")
      
      dunn_res <- subset_data %>%
        dunn_test(as.formula(paste("Value ~", group_column_name)),
                  p.adjust.method = "bonferroni") # Or "holm", "BH", etc.
      
      # Add x and y positions for plotting significance bars
      dunn_res <- dunn_res %>%
        add_xy_position(x = groupcol, fun = "max", step.increase = 0.1)
      
      # Add the 'Measure' column to link results to facets
      dunn_res$Measure <- current_measure
      
      # Combine results for all measures
      pwc_results_for_plot <- rbind(pwc_results_for_plot, dunn_res)
      
      cat("Dunn's Post-Hoc Test Pairwise Results (adjusted p-values):\n")
      print(dunn_res, n = nrow(dunn_res), width=Inf)
      
    } else {
      cat("No significant overall difference for ")
      cat(current_measure)
      cat(". Skipping Dunn's post-hoc test.\n")
    }
  }
  
  # --- 6. Create the Box Plot Matrix with Significance Annotations ---
  cat("\nGenerating box plot matrix...\n")
  
  write.csv(long_data_df, "C:/Users/Gabriel/Documents/long_data_df_new.csv", row.names=FALSE)
  
  plot_matrix <- ggplot(long_data_df, aes_string(x = groupcol, y = "Value", fill = group_column_name)) +
    geom_boxplot() + # Add box plots
    geom_jitter(color = "black", size = 0.8, alpha = 0.6, width = 0.2) + # Add individual data points (jittered)
    labs(
      # x = "Treatment",
      x = as.character(groupcol),
      y = "Value",
      # fill = "Treatment"
      fill = as.character(groupcol)
    ) +
    facet_wrap(~ Measure, scales = "free_y", ncol = 1) + # Arrange measures in rows, allowing free y-scales
    theme_minimal() + # Use a minimal theme for cleaner look
    theme(
      strip.text = element_text(size = 10, face = "bold"), # Style for facet titles
      axis.text.x = element_text(angle = 45, hjust = 1), # Rotate x-axis labels
      legend.position = "bottom", # Place legend at the bottom
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14) # Main title (if added)
    )
  
  write.csv(overall_kruskal_p_values, "C:/Users/Gabriel/Documents/overall_kruskal_p_values_new.csv", row.names=FALSE)
  # Add overall Kruskal-Wallis p-values to each facet
  plot_matrix <- plot_matrix +
    geom_text(
      data = overall_kruskal_p_values,
      aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
      size = 3, fontface = "bold.italic",
      inherit.aes = FALSE # Crucial to not inherit aesthetics from the main plot
    )
  
  # write.csv(pwc_results_for_plot, "C:/Users/Gabriel/Documents/pwc_results_for_plot.csv", row.names=FALSE)
  # Add pairwise significance levels from Dunn's test
  if (nrow(pwc_results_for_plot) > 0) {
    plot_matrix <- plot_matrix +
      stat_pvalue_manual(
        pwc_results_for_plot,
        label = "p.adj.signif", # Display significance level (e.g., *, **, ***)
        hide.ns = TRUE,         # Show non-significant comparisons (ns)
        tip.length = 0.01,
        size = 3.5
      )
  } else {
    cat("\nNo significant pairwise differences to annotate.\n")
  }
  
  # Add a main title for the entire plot matrix
  plot_matrix <- plot_matrix +
    ggtitle("Box Plots of Measures by Treatment with Significance")
  # ggtitle("Box Plots of Measures by Age with Significance")
  
  # Print the arranged plots
  print(plot_matrix)
}





# analyzer(
#   "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/publication_data.csv",
#   c("DMSO", "100nM", "250nM", "1uM"),
#   "treatment",
#   c("stoma_count", "leaf_area_u_sq", "density")
# )

meanalyzer <- function(df_path, ordering, groupcol, valcols) {
  my_data <- read.csv(df_path)
  # my_data <- read.csv("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/section_based_training_07-2025/CLUMPS/ridgeplots_wt/full_EXTENSION.csv")
  
  my_data %<>% mutate(area_u = area*(0.5681818**2))
  
  # treatment_order <- c("DMSO", "100nM", "250nM", "1uM")
  # my_data_raw$treatment <- factor(my_data_raw$treatment, levels = treatment_order, ordered = TRUE)
  # dpg_order <- c("3", "4", "5", "7")
  # my_data[[groupcol]] <- factor(my_data[[groupcol]], levels = ordering, ordered = TRUE)
  
  cat("--- Kruskal-Wallis Test and Post-Hoc Analysis ---\n\n")
  
  # 1. Perform Kruskal-Wallis Test
  # This non-parametric test checks if there's a significant difference
  # in the medians (or distributions) of the 'area' variable across the 'ID' groups.
  kruskal_result <- kruskal.test(area ~ dpg, data = my_data)
  
  cat("Kruskal-Wallis Rank Sum Test Results:\n")
  cat(paste0("  Statistic (Chi-squared): ", round(kruskal_result$statistic, 4), "\n"))
  cat(paste0("  Degrees of Freedom: ", kruskal_result$parameter, "\n"))
  cat(paste0("  P-value: ", format.pval(kruskal_result$p.value, digits = 4), "\n"))
  
  # Interpretation of Kruskal-Wallis
  if (kruskal_result$p.value < 0.05) {
    cat("\nInterpretation: There is a statistically significant difference in 'area' across treatment groups (p < 0.05).\n")
    cat("Proceeding with post-hoc analysis (Dunn's Test) to find specific group differences.\n\n")
    
    # 2. Perform Post-Hoc Analysis (Dunn's Test)
    # Dunn's test is a non-parametric post-hoc test appropriate after a
    # significant Kruskal-Wallis test. It performs pairwise comparisons
    # between groups while adjusting p-values for multiple comparisons.
    # We use the Bonferroni adjustment here, but others (e.g., Holm) are also common.
    # The 'method' parameter determines the p-value adjustment method.
    dunn_result <- dunn.test(
      x = my_data$area,
      g = my_data$dpg,
      method = "bonferroni", # Common adjustment for multiple comparisons
      altp = TRUE             # Display adjusted p-values
    )
    
    cat("Dunn's Test for Multiple Comparisons (Bonferroni p-value adjustment):\n")
    
    # Create a data frame for better organized printing
    # print(length(dunn_result$comparisons))
    # print(length(round(dunn_result$Z)))
    # print(length(format.pval(dunn_result$altP, digits = 4)))
    # print(length(format.pval(dunn_result$altP.adjusted, digits = 4)))
    # print(dunn_result)
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
    group_by(ID, dpg) %>%
    summarise(
      # Use across() to apply mean() to all columns in your vector
      across(
        .cols = all_of(columns_to_mean), # Select the columns
        .fns = ~ mean(.x, na.rm = TRUE), # Define the function
        .names = "mean_{.col}"          # Define a naming convention for new columns
      ),
      n_individuals_in_group = n(),
      .groups = 'drop'
    ) %>%
    ungroup()
  
  
  kruskal_result_mean <- kruskal.test(mean_stomatal_area ~ dpg, data = grouped_data_for_KW)
  
  cat("Kruskal-Wallis Rank Sum Test Results:\n")
  cat(paste0("  Statistic (Chi-squared): ", round(kruskal_result_mean$statistic, 4), "\n"))
  cat(paste0("  Degrees of Freedom: ", kruskal_result_mean$parameter, "\n"))
  cat(paste0("  P-value: ", format.pval(kruskal_result_mean$p.value, digits = 4), "\n"))
  
  # Interpretation of Kruskal-Wallis
  if (kruskal_result_mean$p.value < 0.05) {
    cat("\nInterpretation: There is a statistically significant difference in 'area' across treatment groups (p < 0.05).\n")
    cat("Proceeding with post-hoc analysis (Dunn's Test) to find specific group differences.\n\n")
    
    # 2. Perform Post-Hoc Analysis (Dunn's Test)
    # Dunn's test is a non-parametric post-hoc test appropriate after a
    # significant Kruskal-Wallis test. It performs pairwise comparisons
    # between groups while adjusting p-values for multiple comparisons.
    # We use the Bonferroni adjustment here, but others (e.g., Holm) are also common.
    # The 'method' parameter determines the p-value adjustment method.
    dunn_result_mean <- dunn.test(
      x = grouped_data_for_KW$area,
      g = grouped_data_for_KW$dpg,
      method = "bonferroni", # Common adjustment for multiple comparisons
      altp = TRUE             # Display adjusted p-values
    )
    
    cat("Dunn's Test for Multiple Comparisons (Bonferroni p-value adjustment):\n")
    
    # Create a data frame for better organized printing
    # print(length(dunn_result$comparisons))
    # print(length(round(dunn_result$Z)))
    # print(length(format.pval(dunn_result$altP, digits = 4)))
    # print(length(format.pval(dunn_result$altP.adjusted, digits = 4)))
    # print(dunn_result)
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
}


meanalyzer(
  "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/section_based_training_07-2025/CLUMPS/ridgeplots_wt/full_EXTENSION.csv",
  c("3", "4", "5", "7"),
  "age",
  c("area", "eccentricity", "perimeter")
)

# analyzer(
#   "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/section_based_training_07-2025/CLUMPS/ridgeplots_wt/full_EXTENSION.csv",
#   c("3", "4", "5", "7"),
#   "age",
#   c("stoma_count", "leaf_area_u_sq", "density")
# )

analyzer(
  "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/areas_WT_EXTENSION.csv",
  c("3", "4", "5", "7"),
  "age",
  c("stoma_count", "leaf_area_u_sq", "density")
)