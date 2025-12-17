"""
Field Control Module for NFL Tracking Data

This module computes player influence/field control using a Gaussian influence model.
It also provides batch processing to normalize play direction and append field control
to all files in a directory.

Key Functions:
- append_influence_vectorized: Compute field influence for each player-frame
- process_train_directory: Process all files, normalize direction, append field control
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from joblib import Parallel, delayed
from tqdm import tqdm


# =============================================================================
# Field Rotation (imported functionality, but included here for standalone use)
# =============================================================================

def rotate_field(df):
    """
    Rotate field so all plays have offense attacking from right to left.
    
    This function:
    1. Identifies plays with 'right' direction
    2. Rotates x, y coordinates around field center (60, 26.65)
    3. Adjusts direction (dir) and orientation (o) angles
    
    Args:
        df (pd.DataFrame): Tracking data with columns:
            - play_direction: 'left' or 'right'
            - x, y: position coordinates
            - s: speed (magnitude, unchanged)
            - a: acceleration (magnitude, unchanged)
            - dir: direction angle of motion (degrees)
            - o: orientation angle (degrees)
            
    Returns:
        pd.DataFrame: Data with standardized field direction (all 'left')
    """
    df = df.copy()
    
    # Field dimensions: 120 yards x 53.3 yards
    field_length = 120
    field_width = 53.3
    
    # Identify plays moving right (need rotation)
    right_plays = df['play_direction'] == 'right'
    
    # Rotate position coordinates (180 degree rotation around field center)
    df.loc[right_plays, 'x'] = field_length - df.loc[right_plays, 'x']
    df.loc[right_plays, 'y'] = field_width - df.loc[right_plays, 'y']
    
    # Rotate direction angle (add 180 degrees and normalize to 0-360)
    if 'dir' in df.columns:
        df.loc[right_plays, 'dir'] = (df.loc[right_plays, 'dir'] + 180) % 360
    
    # Rotate orientation angle
    if 'o' in df.columns:
        df.loc[right_plays, 'o'] = (df.loc[right_plays, 'o'] + 180) % 360
    
    # Speed and acceleration magnitudes stay the same (s, a don't change)
    
    # Update play direction to be consistent
    df['play_direction'] = 'left'
    
    # If we have ball landing coordinates, rotate those too
    if 'ball_land_x' in df.columns:
        df.loc[right_plays, 'ball_land_x'] = field_length - df.loc[right_plays, 'ball_land_x']
    if 'ball_land_y' in df.columns:
        df.loc[right_plays, 'ball_land_y'] = field_width - df.loc[right_plays, 'ball_land_y']
    
    return df


# =============================================================================
# Velocity Components
# =============================================================================

def add_velocity_components(df):
    """
    Compute velocity components (vx, vy) from speed and direction.
    
    IMPORTANT: Uses 'dir' (direction of motion) NOT 'o' (orientation).
    
    NFL coordinate system:
    - x: along field length (0-120 yards)
    - y: along field width (0-53.3 yards)
    - dir: angle of motion in degrees (0 = +y direction, 90 = +x direction)
    
    The NFL defines dir as:
    - 0° = moving toward positive y (toward sideline)
    - 90° = moving toward positive x (toward opponent's endzone)
    - 180° = moving toward negative y
    - 270° = moving toward negative x
    
    This means:
    - vx = s * sin(dir)  (90° gives max vx)
    - vy = s * cos(dir)  (0° gives max vy)
    
    Args:
        df: DataFrame with columns 's' (speed) and 'dir' (direction in degrees)
        
    Returns:
        DataFrame with added 'vx' and 'vy' columns
    """
    df = df.copy()
    
    # Use 'dir' (direction of motion), not 'o' (orientation)
    if 'dir' not in df.columns:
        raise ValueError("DataFrame must have 'dir' column for velocity calculation")
    
    dir_rad = np.deg2rad(df['dir'])
    
    # NFL coordinate convention: 0° = +y, 90° = +x
    df['vx'] = df['s'] * np.sin(dir_rad)
    df['vy'] = df['s'] * np.cos(dir_rad)
    
    return df


def add_acceleration_components(df):
    """
    Compute acceleration components (ax, ay) from acceleration magnitude and direction.
    
    Assumes acceleration is in the same direction as velocity (motion direction).
    
    Args:
        df: DataFrame with columns 'a' (acceleration) and 'dir' (direction in degrees)
        
    Returns:
        DataFrame with added 'ax' and 'ay' columns
    """
    df = df.copy()
    
    if 'dir' not in df.columns:
        raise ValueError("DataFrame must have 'dir' column for acceleration calculation")
    
    dir_rad = np.deg2rad(df['dir'])
    
    # Same convention as velocity
    df['ax'] = df['a'] * np.sin(dir_rad)
    df['ay'] = df['a'] * np.cos(dir_rad)
    
    return df


def add_orientation_components(df):
    """
    Compute orientation unit vector components (ox, oy) from orientation angle.
    
    This represents which direction the player is FACING (not moving).
    
    Args:
        df: DataFrame with column 'o' (orientation in degrees)
        
    Returns:
        DataFrame with added 'ox' and 'oy' columns
    """
    df = df.copy()
    
    if 'o' in df.columns:
        o_rad = np.deg2rad(df['o'])
        df['ox'] = np.sin(o_rad)
        df['oy'] = np.cos(o_rad)
    
    return df


# =============================================================================
# Ball Location
# =============================================================================

def append_ball_location(df):
    """
    Add ball location (ball_x, ball_y) based on passer position.
    
    For plays without a passer, uses the backmost offensive player.
    
    Args:
        df: DataFrame with player tracking data
        
    Returns:
        DataFrame with 'ball_x' and 'ball_y' columns added
    """
    df = df.copy()
    
    # Extract passer positions per frame
    passer_mask = df["player_role"].fillna('').str.lower() == "passer"
    passer_positions = (
        df.loc[passer_mask, ["game_id", "play_id", "frame_id", "x", "y"]]
        .rename(columns={"x": "ball_x", "y": "ball_y"})
    )

    # Merge back to all players
    df = df.merge(passer_positions, on=["game_id", "play_id", "frame_id"], how="left")
    
    # For plays without passer, use backmost offensive player (min x for left direction)
    missing_mask = df["ball_x"].isna()
    if missing_mask.any():
        offense_mask = df["player_side"].fillna('').str.lower() == "offense"
        backmost = (
            df[offense_mask]
            .loc[df[offense_mask].groupby(["game_id", "play_id", "frame_id"])["x"].idxmin()]
            [["game_id", "play_id", "frame_id", "x", "y"]]
            .rename(columns={"x": "ball_x_fallback", "y": "ball_y_fallback"})
        )
        df = df.merge(backmost, on=["game_id", "play_id", "frame_id"], how="left")
        df["ball_x"] = df["ball_x"].fillna(df["ball_x_fallback"])
        df["ball_y"] = df["ball_y"].fillna(df["ball_y_fallback"])
        df = df.drop(columns=["ball_x_fallback", "ball_y_fallback"], errors="ignore")
    
    return df


# =============================================================================
# Influence / Field Control Calculation
# =============================================================================

def influence_radius(ball_position, player_position):
    """Example influence radius: bounded by 1–15 yards."""
    dist = np.linalg.norm(ball_position - player_position)
    return max(1.0, min(15.0, dist))


def influence_radius_vec(ball_x, ball_y, player_x, player_y):
    """Vectorized influence radius (bounded by 1–15 yards)."""
    dist = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
    return np.clip(dist, 1.0, 15.0)


def influence_function_single(player_position, player_velocity, ball_position, location):
    """
    Compute influence at one location (ball landing) for one player-frame.
    
    Uses a 2D Gaussian with covariance stretched in the direction of motion.
    """
    theta = np.arctan2(player_velocity[1], player_velocity[0] + 1e-7)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    R_inv = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])

    speed_squared = np.dot(player_velocity, player_velocity)
    srat = speed_squared / (13**2)

    Ri = influence_radius(ball_position, player_position)

    S = np.array([[(1 + srat) * Ri * 0.5, 0],
                  [0, (1 - srat) * Ri * 0.5]])
    Cov = R @ S @ S @ R_inv

    mu = player_position + 0.5 * player_velocity
    gaussian = mvn(mean=mu, cov=Cov)
    return gaussian.pdf(location)


def influence_function_vectorized(df):
    """
    Compute influence at (ball_land_x, ball_land_y) for each player-frame in df.
    Fully vectorized — no loops.
    
    Requires columns:
      ['x', 'y', 'vx', 'vy', 'ball_x', 'ball_y', 'ball_land_x', 'ball_land_y']
      
    Returns:
        DataFrame with 'influence_at_land' and 'influence_at_land_norm' columns
    """
    df = df.copy()
    
    px, py = df["x"].values, df["y"].values
    vx, vy = df["vx"].values, df["vy"].values
    bx, by = df["ball_x"].values, df["ball_y"].values
    lx, ly = df["ball_land_x"].values, df["ball_land_y"].values

    # θ = arctan2(vy, vx) - direction of velocity vector
    theta = np.arctan2(vy, vx + 1e-7)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Influence radius based on distance from ball
    Ri = influence_radius_vec(bx, by, px, py)

    # Speed ratio (normalized by max speed ~13 yards/second)
    speed_sq = vx**2 + vy**2
    srat = speed_sq / (13.0**2)

    # Mean shift - player influence center is slightly ahead of current position
    mu_x = px + 0.5 * vx
    mu_y = py + 0.5 * vy

    # Covariance terms (ellipse stretched in direction of motion)
    # Equivalent to Cov = R @ S @ S @ R_inv
    Sx = (1 + srat) * Ri * 0.5
    Sy = (1 - srat) * Ri * 0.5

    # 2D covariance matrix elements after rotation
    a = (Sx**2) * cos_t**2 + (Sy**2) * sin_t**2
    c = (Sx**2) * sin_t**2 + (Sy**2) * cos_t**2
    b = (Sx**2 - Sy**2) * sin_t * cos_t

    # Inverse covariance for Gaussian PDF
    det = a * c - b**2
    det = np.maximum(det, 1e-10)  # Avoid division by zero
    
    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det))

    # Distance from influence center to ball landing point
    dx = lx - mu_x
    dy = ly - mu_y

    # Gaussian exponent (Mahalanobis distance)
    exponent = -0.5 * (inv_a * dx**2 + 2 * inv_b * dx * dy + inv_c * dy**2)
    pdf = norm_const * np.exp(exponent)

    df["influence_at_land"] = pdf
    
    # Normalize by play so each play's max influence = 1
    df["influence_at_land_norm"] = (
        df.groupby(["game_id", "play_id"])["influence_at_land"]
        .transform(lambda x: x / x.max() if x.max() != 0 else x)
    )

    return df


def append_influence_vectorized(df):
    """
    Add velocity components and compute influence in one call.
    
    This is the main function to call for adding field control.
    
    Args:
        df: DataFrame with tracking data (must have 's', 'dir', 'ball_land_x', 'ball_land_y')
        
    Returns:
        DataFrame with 'vx', 'vy', 'influence_at_land', 'influence_at_land_norm' added
    """
    df = add_velocity_components(df)
    df = influence_function_vectorized(df)
    return df


def append_influence_parallel(df, n_jobs=-1, batch_size=1000):
    """
    Parallelized computation of influence_at_land (slower but useful for debugging).
    """
    df = df.copy()
    df = add_velocity_components(df)
    
    arr = df.to_records(index=False)

    def compute_influence_row(row):
        player_position = np.array([row.x, row.y])
        ball_position = np.array([row.ball_x, row.ball_y])
        player_velocity = np.array([row.vx, row.vy])
        location = np.array([row.ball_land_x, row.ball_land_y])
        return influence_function_single(player_position, player_velocity, ball_position, location)

    influences = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(compute_influence_row)(row)
        for row in tqdm(arr, desc="Computing player influence")
    )

    df["influence_at_land"] = influences
    df["influence_at_land_norm"] = df.groupby(["game_id", "play_id"])["influence_at_land"].transform(
        lambda x: x / x.max() if x.max() != 0 else x
    )

    return df


# =============================================================================
# Batch Processing Function
# =============================================================================

def process_train_directory(
    train_dir: str = 'train',
    output_dir: str = 'with_field_control',
    file_pattern: str = 'input_*.csv'
):
    """
    Process all training files: normalize play direction and append field control.
    
    For each file in train_dir matching file_pattern:
    1. Load the CSV
    2. Rotate field to normalize all plays to 'left' direction
    3. Add velocity components (vx, vy)
    4. Add ball location (ball_x, ball_y)
    5. Compute field control influence
    6. Save to output_dir with same filename
    
    Args:
        train_dir: Directory containing input CSV files
        output_dir: Directory to save processed files
        file_pattern: Glob pattern for input files (default: 'input_*.csv')
        
    Returns:
        List of processed file paths
    """
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(train_path.glob(file_pattern))
    
    if len(input_files) == 0:
        print(f"No files found matching '{file_pattern}' in '{train_dir}'")
        return []
    
    print(f"Found {len(input_files)} files to process")
    processed_files = []
    
    for input_file in tqdm(input_files, desc="Processing files"):
        print(f"\nProcessing: {input_file.name}")
        
        # Load data
        df = pd.read_csv(input_file)
        original_rows = len(df)
        print(f"  Loaded {original_rows} rows")
        
        # Check for required columns
        required_cols = ['play_direction', 'x', 'y', 's', 'dir', 'o']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  WARNING: Missing columns {missing_cols}, skipping file")
            continue
        
        # Step 1: Rotate field to normalize direction
        print("  Rotating field...")
        original_right = (df['play_direction'] == 'right').sum()
        df = rotate_field(df)
        print(f"    Rotated {original_right} 'right' plays to 'left'")
        
        # Step 2: Add velocity and acceleration components
        print("  Adding velocity components...")
        df = add_velocity_components(df)
        df = add_acceleration_components(df)
        df = add_orientation_components(df)
        
        # Step 3: Add ball location
        if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
            print("  Adding ball location...")
            df = append_ball_location(df)
            
            # Step 4: Compute field control influence
            print("  Computing field control...")
            df = influence_function_vectorized(df)
            
            # Report influence stats
            inf_mean = df['influence_at_land'].mean()
            inf_max = df['influence_at_land'].max()
            print(f"    Influence: mean={inf_mean:.6f}, max={inf_max:.6f}")
        else:
            print("  WARNING: No ball_land_x/ball_land_y columns, skipping influence calculation")
        
        # Save processed file
        output_file = output_path / input_file.name
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        
        processed_files.append(output_file)
    
    print(f"\nProcessed {len(processed_files)} files successfully")
    return processed_files


def process_single_file(
    input_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Process a single file: normalize play direction and append field control.
    
    Args:
        input_path: Path to input CSV file
        output_path: Optional path to save processed file
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(input_path)
    
    # Rotate field
    df = rotate_field(df)
    
    # Add velocity components
    df = add_velocity_components(df)
    df = add_acceleration_components(df)
    df = add_orientation_components(df)
    
    # Add ball location and influence if possible
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df = append_ball_location(df)
        df = influence_function_vectorized(df)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NFL tracking data with field control')
    parser.add_argument('--train_dir', type=str, default='train',
                        help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, default='with_field_control',
                        help='Directory to save processed files')
    parser.add_argument('--pattern', type=str, default='input_*.csv',
                        help='Glob pattern for input files')
    parser.add_argument('--single', type=str, default=None,
                        help='Process a single file instead of directory')
    
    args = parser.parse_args()
    
    if args.single:
        # Process single file
        output = args.single.replace('.csv', '_processed.csv')
        process_single_file(args.single, output)
    else:
        # Process directory
        process_train_directory(args.train_dir, args.output_dir, args.pattern)
