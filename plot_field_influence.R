# =============================================================================
# NFL Field Influence Plotting Functions
# =============================================================================
# This script provides functions to visualize field control/influence for NFL
# plays using the Spearman model, compatible with sportyR and gganimate.
#
# Required packages: tidyverse, sportyR, gganimate, MASS (for mvrnorm)
# =============================================================================

library(tidyverse)
library(sportyR)

# =============================================================================
# Core Influence Calculation Functions
# =============================================================================

#' Calculate influence radius based on distance from ball
#' @param ball_x Ball x position
#' @param ball_y Ball y position
#' @param player_x Player x position
#' @param player_y Player y position
#' @return Influence radius (bounded between 1 and 15 yards)
influence_radius <- function(ball_x, ball_y, player_x, player_y) {
  dist <- sqrt((ball_x - player_x)^2 + (ball_y - player_y)^2)
  pmax(1.0, pmin(15.0, dist))
}

#' Calculate player influence at a grid of locations
#' @param player_x Player x position
#' @param player_y Player y position
#' @param player_vx Player velocity x component
#' @param player_vy Player velocity y component
#' @param ball_x Ball x position
#' @param ball_y Ball y position
#' @param grid_x Matrix of x coordinates (from meshgrid)
#' @param grid_y Matrix of y coordinates (from meshgrid)
#' @return Matrix of influence values at each grid point
calculate_player_influence <- function(player_x, player_y, player_vx, player_vy,
                                        ball_x, ball_y, grid_x, grid_y) {
  
  # Direction of motion (Eq. 16)
  theta <- atan2(player_vy, player_vx + 1e-7)
  cos_t <- cos(theta)
  sin_t <- sin(theta)
  
  # Speed ratio (Eq. 18) - normalized by max speed ~13 yards/second
  speed_sq <- player_vx^2 + player_vy^2
  srat <- speed_sq / (13^2)
  
  # Influence radius (Eq. 19)
  Ri <- influence_radius(ball_x, ball_y, player_x, player_y)
  
  # Scaling factors for ellipse
  Sx <- (1 + srat) * Ri * 0.5
  Sy <- (1 - srat) * Ri * 0.5
  
  # Covariance matrix elements after rotation (Eq. 20)
  # Cov = R @ S @ S @ R_inv
  a <- (Sx^2) * cos_t^2 + (Sy^2) * sin_t^2
  c <- (Sx^2) * sin_t^2 + (Sy^2) * cos_t^2
  b <- (Sx^2 - Sy^2) * sin_t * cos_t
  
  # Mean shift - influence center is ahead of player (Eq. 21)
  mu_x <- player_x + 0.5 * player_vx
  mu_y <- player_y + 0.5 * player_vy
  
  # Inverse covariance for Gaussian PDF
  det <- a * c - b^2
  det <- pmax(det, 1e-10)  # Avoid division by zero
  
  inv_a <- c / det
  inv_b <- -b / det
  inv_c <- a / det
  norm_const <- 1.0 / (2 * pi * sqrt(det))
  
  # Distance from influence center to each grid point
  dx <- grid_x - mu_x
  dy <- grid_y - mu_y
  
  # Gaussian PDF (Eq. 12)
  exponent <- -0.5 * (inv_a * dx^2 + 2 * inv_b * dx * dy + inv_c * dy^2)
  pdf <- norm_const * exp(exponent)
  
  # Normalize by player's max influence (Eq. 13)
  pdf / max(pdf, na.rm = TRUE)
}

#' Calculate team field control for a single frame
#' @param frame_data Dataframe containing player data for one frame
#' @param ball_x Ball x position
#' @param ball_y Ball y position
#' @param x_range Vector of x coordinates for grid (default: 0 to 120)
#' @param y_range Vector of y coordinates for grid (default: 0 to 53.3)
#' @param step Grid resolution in yards (default: 1)
#' @return Dataframe with x, y, and control columns for plotting
calculate_field_control <- function(frame_data, ball_x, ball_y,
                                     x_range = c(0, 120),
                                     y_range = c(0, 53.3),
                                     step = 1) {
  
  # Create grid
  x_seq <- seq(x_range[1], x_range[2], by = step)
  y_seq <- seq(y_range[1], y_range[2], by = step)
  grid <- expand.grid(x = x_seq, y = y_seq)
  grid_x <- matrix(grid$x, nrow = length(y_seq), ncol = length(x_seq), byrow = TRUE)
  grid_y <- matrix(grid$y, nrow = length(y_seq), ncol = length(x_seq), byrow = TRUE)
  
  # Initialize influence matrices
  offense_influence <- matrix(0, nrow = length(y_seq), ncol = length(x_seq))
  defense_influence <- matrix(0, nrow = length(y_seq), ncol = length(x_seq))
  
  # Calculate influence for each offensive player
  offense_players <- frame_data %>% filter(tolower(player_side) == "offense")
  for (i in seq_len(nrow(offense_players))) {
    p <- offense_players[i, ]
    if (!is.na(p$vx) && !is.na(p$vy)) {
      player_infl <- calculate_player_influence(
        p$x, p$y, p$vx, p$vy, ball_x, ball_y, grid_x, grid_y
      )
      offense_influence <- offense_influence + player_infl
    }
  }
  
  # Calculate influence for each defensive player
  defense_players <- frame_data %>% filter(tolower(player_side) == "defense")
  for (i in seq_len(nrow(defense_players))) {
    p <- defense_players[i, ]
    if (!is.na(p$vx) && !is.na(p$vy)) {
      player_infl <- calculate_player_influence(
        p$x, p$y, p$vx, p$vy, ball_x, ball_y, grid_x, grid_y
      )
      defense_influence <- defense_influence + player_infl
    }
  }
  
  # Calculate control probability using sigmoid (offense - defense)
  control <- 1 / (1 + exp(-(offense_influence - defense_influence)))
  
  # Convert to dataframe for ggplot
  control_df <- data.frame(
    x = as.vector(grid_x),
    y = as.vector(grid_y),
    control = as.vector(control),
    offense_infl = as.vector(offense_influence),
    defense_infl = as.vector(defense_influence)
  )
  
  return(control_df)
}

# =============================================================================
# Plotting Functions
# =============================================================================

#' Plot field influence for a single frame
#' @param df Dataframe with tracking data (output from field_control.py)
#' @param game_id Game ID to filter
#' @param play_id Play ID to filter
#' @param frame_id Frame ID to plot
#' @param step Grid resolution in yards (default: 1)
#' @param alpha Transparency for influence layer (default: 0.7)
#' @param show_velocities Whether to show velocity arrows (default: TRUE)
#' @param xlims X-axis limits (default: c(10, 110) for main field)
#' @param title Plot title (default: auto-generated)
#' @param subtitle Plot subtitle (default: auto-generated)
#' @param ball_holder_color Color for ball holder/passer (default: "brown")
#' @param ball_holder_label Legend label for ball holder (default: "Ball Holder")
#' @param offense_color Color for offensive players (default: "red")
#' @param offense_label Legend label for offense (default: "Offense")
#' @param defense_color Color for defensive players (default: "blue")
#' @param defense_label Legend label for defense (default: "Defense")
#' @param tracked_color Color for tracked/targeted player (default: "yellow")
#' @param tracked_label Legend label for tracked player (default: "Targeted")
#' @param show_legend Whether to show player legend (default: TRUE)
#' @return ggplot object
plot_field_influence <- function(df, game_id, play_id, frame_id,
                                  step = 1,
                                  alpha = 0.7,
                                  show_velocities = TRUE,
                                  xlims = c(10, 110),
                                  title = NULL,
                                  subtitle = NULL,
                                  ball_holder_color = "brown",
                                  ball_holder_label = "Ball Holder",
                                  offense_color = "red",
                                  offense_label = "Offense",
                                  defense_color = "blue",
                                  defense_label = "Defense",
                                  tracked_color = "yellow",
                                  tracked_label = "Targeted",
                                  show_legend = TRUE) {
  
  # Filter to the specific frame
  frame_data <- df %>%
    filter(game_id == !!game_id,
           play_id == !!play_id,
           frame_id == !!frame_id)
  
  if (nrow(frame_data) == 0) {
    stop("No data found for the specified game_id, play_id, and frame_id")
  }
  
  # Get ball position (from passer or use ball_x/ball_y if available)
  if ("ball_x" %in% names(frame_data) && !all(is.na(frame_data$ball_x))) {
    ball_x <- frame_data$ball_x[1]
    ball_y <- frame_data$ball_y[1]
  } else {
    # Try to find passer position
    passer <- frame_data %>% filter(tolower(player_role) == "passer")
    if (nrow(passer) > 0) {
      ball_x <- passer$x[1]
      ball_y <- passer$y[1]
    } else {
      # Use mean of offensive players as fallback
      offense <- frame_data %>% filter(tolower(player_side) == "offense")
      ball_x <- mean(offense$x, na.rm = TRUE)
      ball_y <- mean(offense$y, na.rm = TRUE)
    }
  }
  
  # Calculate field control
  control_df <- calculate_field_control(
    frame_data, ball_x, ball_y,
    x_range = c(0, 120),
    y_range = c(0, 53.3),
    step = step
  )
  
  # Set up field parameters
  field_params <- list(
    field_apron = "springgreen3",
    field_border = "springgreen3",
    offensive_endzone = "springgreen3",
    defensive_endzone = "springgreen3",
    offensive_half = "springgreen3",
    defensive_half = "springgreen3"
  )
  
  # Create base plot with field
  p <- geom_football(
    league = "nfl",
    display_range = "in_bounds_only",
    x_trans = 60,
    y_trans = 26.6667,
    xlims = xlims,
    color_updates = field_params
  )
  
  # Add influence heatmap
  p <- p +
    geom_tile(data = control_df,
              aes(x = x, y = y, fill = control),
              alpha = alpha) +
    scale_fill_gradient2(
      low = defense_color,
      mid = "white",
      high = offense_color,
      midpoint = 0.5,
      limits = c(0, 1),
      name = paste0(offense_label, "\nControl")
    )
  
  # Prepare player data with assigned colors
  player_data <- frame_data %>%
    mutate(
      player_type = case_when(
        tolower(player_role) == "targeted receiver" ~ tracked_label,
        tolower(player_role) == "passer" ~ ball_holder_label,
        tolower(player_side) == "offense" ~ offense_label,
        tolower(player_side) == "defense" ~ defense_label,
        TRUE ~ "Other"
      ),
      pt_color = case_when(
        tolower(player_role) == "targeted receiver" ~ tracked_color,
        tolower(player_role) == "passer" ~ ball_holder_color,
        tolower(player_side) == "offense" ~ offense_color,
        tolower(player_side) == "defense" ~ defense_color,
        TRUE ~ "gray"
      )
    )
  
  # Add players (direct colors, no fill aesthetic)
  p <- p +
    geom_point(data = player_data,
               aes(x = x, y = y),
               fill = player_data$pt_color,
               color = "black",
               size = 6,
               shape = 21,
               stroke = 1.5)
  
  # Add manual legend for players using color aesthetic on dummy off-screen points
  if (show_legend) {
    legend_data <- data.frame(
      player_type = c(ball_holder_label, offense_label, defense_label, tracked_label),
      x = rep(-100, 4),
      y = rep(-100, 4)
    )
    
    p <- p +
      geom_point(data = legend_data,
                 aes(x = x, y = y, color = player_type),
                 size = 6, shape = 16) +
      scale_color_manual(
        values = setNames(
          c(ball_holder_color, offense_color, defense_color, tracked_color),
          c(ball_holder_label, offense_label, defense_label, tracked_label)
        ),
        name = "Players"
      )
  }
  
  # Add velocity arrows if requested
  if (show_velocities && "vx" %in% names(player_data)) {
    arrow_scale <- 0.5
    p <- p +
      geom_segment(data = player_data %>% filter(!is.na(vx), !is.na(vy)),
                   aes(x = x, y = y,
                       xend = x + vx * arrow_scale,
                       yend = y + vy * arrow_scale),
                   arrow = arrow(length = unit(0.15, "inches")),
                   color = "black",
                   linewidth = 0.8)
  }
  
  # Add ball position marker
  p <- p +
    annotate("point", x = ball_x, y = ball_y,
             shape = 21, size = 4, fill = ball_holder_color, color = "white", stroke = 2)
  
  # Set default title and subtitle if not provided
  if (is.null(title)) {
    title <- sprintf("Field Influence: Game %s, Play %s, Frame %s",
                     game_id, play_id, frame_id)
  }
  if (is.null(subtitle)) {
    subtitle <- sprintf("%s = %s Control, %s = %s Control",
                        offense_color, offense_label, defense_color, defense_label)
  }
  
  # Add title
  p <- p +
    labs(
      title = title,
      subtitle = subtitle
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      legend.position = if (show_legend) "right" else "none"
    )
  
  return(p)
}

#' Create animated field influence for a full play
#' @param df Dataframe with tracking data
#' @param game_id Game ID to filter
#' @param play_id Play ID to filter
#' @param step Grid resolution in yards (default: 2 for speed)
#' @param fps Frames per second for animation (default: 10)
#' @param title Plot title (default: auto-generated, use {frame_time} for frame number)
#' @param subtitle Plot subtitle (default: "Frame: {frame_time}")
#' @param ball_holder_color Color for ball holder/passer (default: "brown")
#' @param ball_holder_label Legend label for ball holder (default: "Ball Holder")
#' @param offense_color Color for offensive players (default: "red")
#' @param offense_label Legend label for offense (default: "Offense")
#' @param defense_color Color for defensive players (default: "blue")
#' @param defense_label Legend label for defense (default: "Defense")
#' @param tracked_color Color for tracked/targeted player (default: "yellow")
#' @param tracked_label Legend label for tracked player (default: "Targeted")
#' @param show_legend Whether to show player legend (default: TRUE)
#' @param xlims X-axis limits (default: c(10, 110))
#' @param alpha Transparency for influence layer (default: 0.7)
#' @return gganimate object
plot_field_influence_animated <- function(df, game_id, play_id,
                                           step = 2,
                                           fps = 10,
                                           title = NULL,
                                           subtitle = "Frame: {frame_time}",
                                           ball_holder_color = "brown",
                                           ball_holder_label = "Ball Holder",
                                           offense_color = "red",
                                           offense_label = "Offense",
                                           defense_color = "blue",
                                           defense_label = "Defense",
                                           tracked_color = "yellow",
                                           tracked_label = "Targeted",
                                           show_legend = TRUE,
                                           xlims = c(10, 110),
                                           alpha = 0.7) {
  library(gganimate)
  
  # Filter to the specific play
  play_data <- df %>%
    filter(game_id == !!game_id,
           play_id == !!play_id)
  
  if (nrow(play_data) == 0) {
    stop("No data found for the specified game_id and play_id")
  }
  
  # Get all unique frames
  frames <- sort(unique(play_data$frame_id))
  
  # Calculate field control for each frame
  all_control <- map_dfr(frames, function(fid) {
    frame_data <- play_data %>% filter(frame_id == fid)
    
    # Get ball position
    if ("ball_x" %in% names(frame_data) && !all(is.na(frame_data$ball_x))) {
      ball_x <- frame_data$ball_x[1]
      ball_y <- frame_data$ball_y[1]
    } else {
      passer <- frame_data %>% filter(tolower(player_role) == "passer")
      if (nrow(passer) > 0) {
        ball_x <- passer$x[1]
        ball_y <- passer$y[1]
      } else {
        offense <- frame_data %>% filter(tolower(player_side) == "offense")
        ball_x <- mean(offense$x, na.rm = TRUE)
        ball_y <- mean(offense$y, na.rm = TRUE)
      }
    }
    
    control_df <- calculate_field_control(
      frame_data, ball_x, ball_y,
      step = step
    )
    control_df$frame_id <- fid
    control_df
  })
  
  # Prepare player data with assigned colors
  player_data <- play_data %>%
    mutate(
      player_type = case_when(
        tolower(player_role) == "targeted receiver" ~ tracked_label,
        tolower(player_role) == "passer" ~ ball_holder_label,
        tolower(player_side) == "offense" ~ offense_label,
        tolower(player_side) == "defense" ~ defense_label,
        TRUE ~ "Other"
      ),
      pt_color = case_when(
        tolower(player_role) == "targeted receiver" ~ tracked_color,
        tolower(player_role) == "passer" ~ ball_holder_color,
        tolower(player_side) == "offense" ~ offense_color,
        tolower(player_side) == "defense" ~ defense_color,
        TRUE ~ "gray"
      )
    )
  
  # Set up field
  field_params <- list(
    field_apron = "springgreen3",
    field_border = "springgreen3",
    offensive_endzone = "springgreen3",
    defensive_endzone = "springgreen3",
    offensive_half = "springgreen3",
    defensive_half = "springgreen3"
  )
  
  # Set default title if not provided
  if (is.null(title)) {
    title <- sprintf("Field Influence: Game %s, Play %s", game_id, play_id)
  }
  
  # Create base animated plot
  p <- geom_football(
    league = "nfl",
    display_range = "in_bounds_only",
    x_trans = 60,
    y_trans = 26.6667,
    xlims = xlims,
    color_updates = field_params
  ) +
    geom_tile(data = all_control,
              aes(x = x, y = y, fill = control),
              alpha = alpha) +
    scale_fill_gradient2(
      low = defense_color, 
      mid = "white", 
      high = offense_color,
      midpoint = 0.5, 
      limits = c(0, 1),
      name = paste0(offense_label, "\nControl")
    ) +
    geom_point(data = player_data,
               aes(x = x, y = y),
               fill = player_data$pt_color,
               color = "black",
               size = 5,
               shape = 21,
               stroke = 1.5)
  
  # Add manual legend for players using color aesthetic on dummy off-screen points
  if (show_legend) {
    legend_data <- data.frame(
      player_type = rep(c(ball_holder_label, offense_label, defense_label, tracked_label), length(frames)),
      frame_id = rep(frames, each = 4),
      x = rep(-100, 4 * length(frames)),
      y = rep(-100, 4 * length(frames))
    )
    
    p <- p +
      geom_point(data = legend_data,
                 aes(x = x, y = y, color = player_type),
                 size = 5, shape = 16) +
      scale_color_manual(
        values = setNames(
          c(ball_holder_color, offense_color, defense_color, tracked_color),
          c(ball_holder_label, offense_label, defense_label, tracked_label)
        ),
        name = "Players"
      )
  }
  
  p <- p +
    labs(
      title = title,
      subtitle = subtitle
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      legend.position = if (show_legend) "right" else "none"
    ) +
    transition_time(frame_id) +
    ease_aes('linear')
  
  return(p)
}

# =============================================================================
# Example Usage
# =============================================================================

# # Load your processed data (output from field_control.py)
# sample <- read_csv("with_field_control/input_2023_w01.csv")
#
# # Plot single frame
# p <- plot_field_influence(
#   df = sample,
#   game_id = 2023091001,
#   play_id = 3216,
#   frame_id = 10,
#   step = 1,
#   offense_color = "red",
#   offense_label = "Chiefs",
#   defense_color = "blue",
#   defense_label = "Ravens"
# )
# print(p)
#
# # Save plot
# ggsave("field_influence.png", p, width = 14, height = 8)
#
# # Create animation
# anim <- plot_field_influence_animated(
#   df = sample,
#   game_id = 2023091001,
#   play_id = 3216,
#   step = 2,
#   fps = 10
# )
# anim_save("field_influence.gif", anim, fps = 10)
