library(ggplot2)


library(magrittr)
library(dplyr)
library(ggridges)

data_df <- read.csv("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/clumps/AZD/full.csv")
data_df %<>% mutate(area_u = area*(0.5681818**2))
data_df$ID <- as.factor(data_df$ID)

p_ointcloud_axes <- ggplot(data_df, aes(x=axis_minor_length, y=axis_major_length)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
  geom_abline(intercept = 0, slope = 1.618, linetype = "solid", color = "yellow") +
  geom_abline(intercept = 0, slope = 2.5, linetype = "dashed", color = "black")

# print(p_ointcloud_axes)

ptcld_DMSO <- ggplot(data_df[data_df$treatment == "DMSO",], aes(x=axis_minor_length, y=axis_major_length, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for DMSO",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 60) +
  ylim(0, 100) +
  theme_minimal()
ptcld_A <- ggplot(data_df[data_df$treatment == "100nM",], aes(x=axis_minor_length, y=axis_major_length, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 100uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 60) +
  ylim(0, 100) +
  theme_minimal()
ptcld_B <- ggplot(data_df[data_df$treatment == "250nM",], aes(x=axis_minor_length, y=axis_major_length, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 250uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 60) +
  ylim(0, 100) +
  theme_minimal()
ptcld_C <- ggplot(data_df[data_df$treatment == "1uM",], aes(x=axis_minor_length, y=axis_major_length, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 1uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 60) +
  ylim(0, 100) +
  theme_minimal()
ptcld_A
ptcld_B
ptcld_C

# print(ptcld_DMSO)
# print(ptcld_A)
# print(ptcld_B)
# print(ptcld_C)

ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_DMSO_lengths.svg",
  plot = ptcld_DMSO,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_100nM_lengths.svg",
  plot = ptcld_A,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_250nM_lengths.svg",
  plot = ptcld_B,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_1uM_lengths.svg",
  plot = ptcld_C,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
# 
# 
ptcld_DMSO_warp <- ggplot(data_df[data_df$treatment == "DMSO",], aes(x=area, y=eccentricity, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for DMSO",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 2700) +
  ylim(0, 1) +
  theme_minimal()
ptcld_A_warp <- ggplot(data_df[data_df$treatment == "100nM",], aes(x=area, y=eccentricity, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 100uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 2700) +
  ylim(0, 1) +
  theme_minimal()
ptcld_B_warp <- ggplot(data_df[data_df$treatment == "250nM",], aes(x=area, y=eccentricity, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 250uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 2700) +
  ylim(0, 1) +
  theme_minimal()
ptcld_C_warp <- ggplot(data_df[data_df$treatment == "1uM",], aes(x=area, y=eccentricity, fill = treatment)) +
  stat_density2d(geom = "tile", aes(fill = after_stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "black", high = "green") + # Customize color scale
  labs(title = "Density Heatmap for 1uM",
       x = "X-axis", y = "Y-axis", fill = "Density") +
  xlim(0, 2700) +
  ylim(0, 1) +
  theme_minimal()
# ptcld_A
# ptcld_B
# ptcld_C

print(ptcld_DMSO_warp)
print(ptcld_A_warp)
print(ptcld_B_warp)
print(ptcld_C_warp)
# 
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_DMSO_warp.svg",
  plot = ptcld_DMSO_warp,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_100nM_warp.svg",
  plot = ptcld_A_warp,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_250nM_warp.svg",
  plot = ptcld_B_warp,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ptcld_1uM_warp.svg",
  plot = ptcld_C_warp,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)

# bxplt_minorlength <- ggplot(data_df, aes(x = factor(treatment, levels = c("DMSO", "100nM", "250nM", "1uM")), y = axis_minor_length*(0.5681818))) +
#   geom_violin(width = 0.8) +
#   geom_boxplot(fill = "white", width=0.2) +
#   labs(title = "Boxplot of stomatal width by treatment",
#        x = "Treatment",
#        y = "Stomatal Width (microns)") +
#   theme_minimal()
# 
# print(bxplt_minorlength)
# 
# bxplt_majorlength <- ggplot(data_df, aes(x = factor(treatment, levels = c("DMSO", "100nM", "250nM", "1uM")), y = axis_major_length*(0.5681818))) +
#   geom_violin(width = 0.8) +
#   geom_boxplot(fill = "white", width=0.2) +
#   labs(title = "Boxplot of stomatal length by treatment",
#        x = "Treatment",
#        y = "Stomatal Length (microns)") +
#   theme_minimal()
# 
# print(bxplt_majorlength)
# 
# bxplt_area <- ggplot(data_df, aes(x = factor(treatment, levels = c("DMSO", "100nM", "250nM", "1uM")), y = area*(0.5681818**2))) +
#   geom_violin(width = 0.8) +
#   geom_boxplot(fill = "white", width=0.2) +
#   labs(title = "Boxplot of stomatal area by treatment",
#        x = "Treatment",
#        y = "Stomatal Area (square microns)") +
#   theme_minimal()
# 
# print(bxplt_area)
# 
# 
# bxplt_perimeter <- ggplot(data_df, aes(x = factor(treatment, levels = c("DMSO", "100nM", "250nM", "1uM")), y = perimeter*(0.5681818))) +
#   geom_violin(width = 0.8) +
#   geom_boxplot(fill = "white", width=0.2) +
#   labs(title = "Boxplot of stomatal perimeter by treatment",
#        x = "Treatment",
#        y = "Stomatal Perimeter (microns)") +
#   theme_minimal()
# 
# print(bxplt_perimeter)

#-----------------

bxplt_minorlength_ridge <- ggplot(data_df, aes(x = axis_minor_length * (0.5681818), 
                                               y = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))), 
                                               fill = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))))) +
  geom_density_ridges(alpha = 0.8, quantile_lines=TRUE) +
  scale_fill_manual(values = c("DMSO" = "#000000", "100nM" = "#77FF00", "250nM" = "#BBD800", "1uM"="#FFA500")) +
  labs(title = "Ridgeplot of Stomatal Width by Treatment",
       x = "Stomatal Width (microns)",
       y = "Treatment") +
  xlim(0, 40) +
  theme_minimal() +
  theme(legend.position = "none") # Remove legend as treatment is on Y-axis

print(bxplt_minorlength_ridge)

ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ridge_AZD_width.svg",
  plot = bxplt_minorlength_ridge,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)

bxplt_majorlength_ridge <- ggplot(data_df, aes(x = axis_major_length * (0.5681818), 
                                               y = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))), 
                                               fill = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))))) +
  scale_fill_manual(values = c("DMSO" = "#000000", "100nM" = "#77FF00", "250nM" = "#BBD800", "1uM"="#FFA500")) +
  geom_density_ridges(alpha = 0.8, quantile_lines=TRUE) +
  labs(title = "Ridgeplot of Stomatal Length by Treatment",
       x = "Stomatal Length (microns)",
       y = "Treatment") +
  xlim(0, 40) +
  theme_minimal() +
  theme(legend.position = "none")

print(bxplt_majorlength_ridge)

ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ridge_AZD_length.svg",
  plot = bxplt_majorlength_ridge,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)

bxplt_area_ridge <- ggplot(data_df, aes(x = area * (0.5681818**2), 
                                        y = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))), 
                                        fill = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))))) +
  scale_fill_manual(values = c("DMSO" = "#000000", "100nM" = "#77FF00", "250nM" = "#BBD800", "1uM"="#FFA500")) +
  geom_density_ridges(alpha = 0.8, quantile_lines=TRUE) +
  labs(title = "Ridgeplot of Stomatal Area by Treatment",
       x = "Stomatal Area (square microns)",
       y = "Treatment") +
  xlim(0, 900) +
  theme_minimal() +
  theme(legend.position = "none")

print(bxplt_area_ridge)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ridge_AZD_area.svg",
  plot = bxplt_area_ridge,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)

bxplt_perimeter_ridge <- ggplot(data_df, aes(x = perimeter * (0.5681818), 
                                        y = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))), 
                                        fill = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))))) +
  scale_fill_manual(values = c("DMSO" = "#000000", "100nM" = "#77FF00", "250nM" = "#BBD800", "1uM"="#FFA500")) +
  geom_density_ridges(alpha = 0.8, quantile_lines=TRUE) +
  labs(title = "Ridgeplot of Stomatal Perimeter by Treatment",
       x = "Stomatal Perimeter (microns)",
       y = "Treatment") +
  xlim(0, 125) +
  theme_minimal() +
  theme(legend.position = "none")

print(bxplt_perimeter_ridge)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ridge_AZD_perimeter.svg",
  plot = bxplt_perimeter_ridge,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)

bxplt_perimeter_ridge <- ggplot(data_df, aes(x = perimeter * (0.5681818), 
                                        y = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))), 
                                        fill = factor(treatment, levels = rev(c("DMSO", "100nM", "250nM", "1uM"))))) +
  scale_fill_manual(values = c("DMSO" = "#000000", "100nM" = "#77FF00", "250nM" = "#BBD800", "1uM"="#FFA500")) +
  geom_density_ridges(alpha = 0.8, quantile_lines=TRUE) +
  labs(title = "Ridgeplot of Stomatal Perimeter by Treatment",
       x = "Stomatal Perimeter (microns)",
       y = "Treatment") +
  xlim(0, 125) +
  theme_minimal() +
  theme(legend.position = "none")

print(bxplt_perimeter_ridge)
ggsave(
  filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/ridge_AZD_perimeter.svg",
  plot = bxplt_perimeter_ridge,
  width = 6,
  height = 6,
  units = "in",
  dpi = 300
)