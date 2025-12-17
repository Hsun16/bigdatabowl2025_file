"""
Defender Field Control Gain Metrics

This module computes metrics to measure how quickly defenders close down space
after the ball is thrown, based on their gain of field influence at the ball
landing point.

Key Functions:
- load_and_join_input_output: Join input and output tracking data
- compute_influence_for_combined: Calculate influence for all frames
- calculate_influence_gain_rate: Compute average influence gain per player

Updates:
- Uses last input frame's ball position and player velocity for output frames
- Uses z-score normalized influence for gain rate calculations
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional

# Import from field_control.py (assuming it's in the same directory)
from field_control import (
    add_velocity_components,
    influence_function_vectorized,
    influence_radius_vec
)


# =============================================================================
# Data Loading and Joining
# =============================================================================

def load_and_join_input_output(
    input_dir: str = 'with_field_control',
    output_dir: str = 'train',
    input_pattern: str = 'input_2023_w*.csv',
    output_pattern: str = 'output_2023_w*.csv'
) -> pd.DataFrame:
    """
    Load and join input and output tracking data.
    
    Steps:
    1. Filter input data to include only defenders who are players to predict
    2. Join input and output data on (game_id, play_id, nfl_id)
    3. Adjust frame_id in output data (add max input frame_id per play)
    4. Carry forward last input frame's ball position and velocity to output frames
    
    Args:
        input_dir: Directory containing processed input files (with field control)
        output_dir: Directory containing output files (post-throw tracking)
        input_pattern: Glob pattern for input files
        output_pattern: Glob pattern for output files
        
    Returns:
        Combined DataFrame with all frames (pre and post throw)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find matching file pairs
    input_files = sorted(input_path.glob(input_pattern))
    output_files = sorted(output_path.glob(output_pattern))
    
    print(f"Found {len(input_files)} input files and {len(output_files)} output files")
    
    all_combined = []
    
    for input_file in tqdm(input_files, desc="Processing file pairs"):
        # Find corresponding output file
        # input: input_2023_w01.csv -> output: output_2023_w01.csv
        week_suffix = input_file.name.replace('input_', '').replace('.csv', '')
        output_file = output_path / f"output_{week_suffix}.csv"
        
        if not output_file.exists():
            print(f"  Warning: No matching output file for {input_file.name}")
            continue
        
        # Load data
        input_df = pd.read_csv(input_file)
        output_df = pd.read_csv(output_file)
        
        # Step 1: Filter input to defenders who are players to predict
        if 'player_to_predict' in input_df.columns:
            defenders_to_predict = input_df[
                (input_df['player_side'].str.lower() == 'defense') &
                (input_df['player_to_predict'] == True)
            ].copy()
        else:
            # If no player_to_predict column, use all defenders that appear in output
            output_players = output_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
            defenders_to_predict = input_df[
                input_df['player_side'].str.lower() == 'defense'
            ].merge(output_players, on=['game_id', 'play_id', 'nfl_id'], how='inner')
        
        if len(defenders_to_predict) == 0:
            continue
        
        # Mark data source
        defenders_to_predict['data_source'] = 'input'
        
        # Step 2: Get max frame_id per play from input data
        max_frames = defenders_to_predict.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
        max_frames.columns = ['game_id', 'play_id', 'max_input_frame']
        
        # Step 3: Get last input frame info per player (ball position, velocity)
        # This will be used for computing influence in output frames
        last_input_frame = defenders_to_predict.sort_values('frame_id').groupby(
            ['game_id', 'play_id', 'nfl_id']
        ).last().reset_index()
        
        # Select columns to carry forward
        carry_forward_cols = ['game_id', 'play_id', 'nfl_id']
        optional_cols = ['ball_x', 'ball_y', 'vx', 'vy', 'ball_land_x', 'ball_land_y', 
                         'player_side', 'player_role', 'player_name', 'player_position']
        for col in optional_cols:
            if col in last_input_frame.columns:
                carry_forward_cols.append(col)
        
        last_input_info = last_input_frame[carry_forward_cols].copy()
        # Rename to indicate these are from last input frame
        rename_cols = {col: f'{col}_last_input' for col in optional_cols if col in last_input_info.columns}
        last_input_info = last_input_info.rename(columns=rename_cols)
        
        # Step 4: Filter output to only include players in our defender list
        defender_keys = defenders_to_predict[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
        output_filtered = output_df.merge(defender_keys, on=['game_id', 'play_id', 'nfl_id'], how='inner')
        
        if len(output_filtered) == 0:
            continue
        
        # Adjust frame_id in output (add max input frame to make continuous)
        output_filtered = output_filtered.merge(max_frames, on=['game_id', 'play_id'], how='left')
        output_filtered['frame_id'] = output_filtered['frame_id'] + output_filtered['max_input_frame']
        output_filtered = output_filtered.drop(columns=['max_input_frame'])
        output_filtered['data_source'] = 'output'
        
        # Merge last input frame info to output
        output_filtered = output_filtered.merge(last_input_info, on=['game_id', 'play_id', 'nfl_id'], how='left')
        
        # Copy carried forward columns to standard column names for output frames
        for col in optional_cols:
            last_input_col = f'{col}_last_input'
            if last_input_col in output_filtered.columns:
                if col not in output_filtered.columns:
                    output_filtered[col] = output_filtered[last_input_col]
                else:
                    # Fill NaN values with last input values
                    output_filtered[col] = output_filtered[col].fillna(output_filtered[last_input_col])
        
        # Drop the _last_input columns
        drop_cols = [c for c in output_filtered.columns if c.endswith('_last_input')]
        output_filtered = output_filtered.drop(columns=drop_cols, errors='ignore')
        
        # Combine input and output
        # Select common columns
        common_cols = list(set(defenders_to_predict.columns) & set(output_filtered.columns))
        
        combined = pd.concat([
            defenders_to_predict[common_cols],
            output_filtered[common_cols]
        ], ignore_index=True)
        
        all_combined.append(combined)
    
    if len(all_combined) == 0:
        print("No data found!")
        return pd.DataFrame()
    
    result = pd.concat(all_combined, ignore_index=True)
    print(f"Combined data: {len(result)} rows")
    
    return result


def compute_influence_for_output_frames(
    combined_df: pd.DataFrame,
    ball_x_col: str = 'ball_land_x',
    ball_y_col: str = 'ball_land_y'
) -> pd.DataFrame:
    """
    Compute influence at ball landing point for all frames.
    
    For output frames, uses the LAST INPUT FRAME's ball position and player velocity
    (not estimated from position differences).
    
    Args:
        combined_df: Combined input/output DataFrame
        ball_x_col: Column name for ball landing x coordinate
        ball_y_col: Column name for ball landing y coordinate
        
    Returns:
        DataFrame with influence_at_land computed for all frames
    """
    df = combined_df.copy()
    
    # Compute influence for all frames using the velocity columns
    # (For output frames, vx/vy should already be filled from last input frame)
    
    if 'vx' not in df.columns or 'vy' not in df.columns:
        raise ValueError("DataFrame must have 'vx' and 'vy' columns")
    
    if ball_x_col not in df.columns or ball_y_col not in df.columns:
        raise ValueError(f"DataFrame must have '{ball_x_col}' and '{ball_y_col}' columns")
    
    # Compute influence for all rows
    df = compute_influence_direct(df, ball_x_col, ball_y_col)
    
    return df


def compute_influence_direct(
    df: pd.DataFrame,
    ball_land_x_col: str = 'ball_land_x',
    ball_land_y_col: str = 'ball_land_y'
) -> pd.DataFrame:
    """
    Compute influence at ball landing point directly using vectorized method.
    
    Args:
        df: DataFrame with x, y, vx, vy, ball_land_x, ball_land_y columns
        ball_land_x_col: Column name for ball landing x
        ball_land_y_col: Column name for ball landing y
        
    Returns:
        DataFrame with influence_at_land column added
    """
    df = df.copy()
    
    # Get required values
    px, py = df['x'].values, df['y'].values
    vx, vy = df['vx'].values, df['vy'].values
    lx, ly = df[ball_land_x_col].values, df[ball_land_y_col].values
    
    # Use player position as ball position for influence radius calculation
    bx, by = px, py  # Simplified - could use actual ball position if available
    
    # θ = arctan2(vy, vx) - direction of velocity vector
    theta = np.arctan2(vy, vx + 1e-7)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # Influence radius based on distance from ball
    Ri = influence_radius_vec(bx, by, px, py)
    # For self-distance, set minimum radius
    Ri = np.maximum(Ri, 4.0)
    
    # Speed ratio (normalized by max speed ~13 yards/second)
    speed_sq = vx**2 + vy**2
    srat = speed_sq / (13.0**2)
    
    # Mean shift - player influence center is slightly ahead of current position
    mu_x = px + 0.5 * vx
    mu_y = py + 0.5 * vy
    
    # Covariance terms (ellipse stretched in direction of motion)
    Sx = (1 + srat) * Ri * 0.5
    Sy = (1 - srat) * Ri * 0.5
    
    # 2D covariance matrix elements after rotation
    a = (Sx**2) * cos_t**2 + (Sy**2) * sin_t**2
    c = (Sx**2) * sin_t**2 + (Sy**2) * cos_t**2
    b = (Sx**2 - Sy**2) * sin_t * cos_t
    
    # Inverse covariance for Gaussian PDF
    det = a * c - b**2
    det = np.maximum(det, 1e-10)
    
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
    
    df['influence_at_land'] = pdf
    
    return df


def compute_zscore_normalized_influence(
    df: pd.DataFrame,
    group_cols: list = ['game_id', 'play_id']
) -> pd.DataFrame:
    """
    Compute z-score normalized influence within each play.
    
    Z-score normalization: (x - mean) / std
    
    This normalizes influence relative to other players in the same play,
    making it easier to compare across different plays.
    
    Args:
        df: DataFrame with influence_at_land column
        group_cols: Columns to group by for normalization (default: per play)
        
    Returns:
        DataFrame with influence_zscore column added
    """
    df = df.copy()
    
    if 'influence_at_land' not in df.columns:
        raise ValueError("DataFrame must have 'influence_at_land' column")
    
    # Compute z-score within each group
    def zscore(x):
        mean = x.mean()
        std = x.std()
        if std == 0 or np.isnan(std):
            return x - mean  # Just center if no variance
        return (x - mean) / std
    
    df['influence_zscore'] = df.groupby(group_cols)['influence_at_land'].transform(zscore)
    
    return df


# =============================================================================
# Influence Gain Rate Calculation
# =============================================================================

def calculate_influence_gain_rate(
    combined_df: pd.DataFrame,
    fps: float = 10.0,
    use_zscore: bool = True
) -> pd.DataFrame:
    """
    Calculate average gain of field control at ball landing point per player.
    
    For each player in each play:
    - Get influence at last input frame (just before throw)
    - Get influence at last output frame (end of ball flight)
    - Compute: (final_influence - initial_influence) / (num_output_frames / fps)
    
    This gives the rate of influence gain in influence units per second.
    
    Args:
        combined_df: DataFrame with influence computed for all frames
        fps: Frames per second (default 10)
        use_zscore: If True, use z-score normalized influence (default True)
        
    Returns:
        DataFrame with per-player influence gain metrics:
        - game_id, play_id, nfl_id: identifiers
        - initial_influence: influence at last pre-throw frame
        - final_influence: influence at last post-throw frame
        - influence_change: final - initial
        - num_output_frames: number of frames after throw
        - time_elapsed: num_output_frames / fps
        - influence_gain_rate: influence_change / time_elapsed
    """
    df = combined_df.copy()
    
    # Compute z-score normalized influence if requested
    if use_zscore:
        if 'influence_zscore' not in df.columns:
            df = compute_zscore_normalized_influence(df)
        influence_col = 'influence_zscore'
    else:
        influence_col = 'influence_at_land'
    
    # Ensure we have required columns
    if influence_col not in df.columns:
        raise ValueError(f"DataFrame must have '{influence_col}' column. "
                        "Run compute_influence_for_output_frames first.")
    
    results = []
    
    # Group by player in each play
    for (game_id, play_id, nfl_id), player_data in df.groupby(['game_id', 'play_id', 'nfl_id']):
        
        # Separate input and output frames
        input_frames = player_data[player_data['data_source'] == 'input'].sort_values('frame_id')
        output_frames = player_data[player_data['data_source'] == 'output'].sort_values('frame_id')
        
        if len(input_frames) == 0 or len(output_frames) == 0:
            continue
        
        # Get initial influence (last input frame = just before throw)
        initial_influence = input_frames[influence_col].iloc[-1]
        
        # Get final influence (last output frame)
        final_influence = output_frames[influence_col].iloc[-1]
        
        # Skip if either is NaN
        if np.isnan(initial_influence) or np.isnan(final_influence):
            continue
        
        # Number of output frames
        num_output_frames = len(output_frames)
        
        # Time elapsed in seconds
        time_elapsed = num_output_frames / fps
        
        # Influence change
        influence_change = final_influence - initial_influence
        
        # Influence gain rate (per second)
        influence_gain_rate = influence_change / time_elapsed if time_elapsed > 0 else 0
        
        # Get player info if available
        player_info = {
            'game_id': game_id,
            'play_id': play_id,
            'nfl_id': nfl_id,
            'initial_influence': initial_influence,
            'final_influence': final_influence,
            'influence_change': influence_change,
            'num_output_frames': num_output_frames,
            'time_elapsed': time_elapsed,
            'influence_gain_rate': influence_gain_rate,
            'normalized_method': 'zscore' if use_zscore else 'raw'
        }
        
        # Add optional player info
        for col in ['player_name', 'player_position', 'player_role', 'player_side']:
            if col in player_data.columns:
                val = player_data[col].dropna().iloc[0] if len(player_data[col].dropna()) > 0 else None
                player_info[col] = val
        
        results.append(player_info)
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        # Sort by influence gain rate (defenders closing down = negative for offense perspective)
        result_df = result_df.sort_values('influence_gain_rate', ascending=False)
    
    return result_df


def calculate_player_average_gain_rate(
    influence_gain_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate average influence gain rate across all plays for each player.
    
    Args:
        influence_gain_df: Output from calculate_influence_gain_rate
        
    Returns:
        DataFrame with per-player averages:
        - nfl_id: player identifier
        - player_name, player_position: if available
        - num_plays: number of plays
        - avg_influence_gain_rate: mean gain rate across plays
        - std_influence_gain_rate: standard deviation
        - avg_initial_influence: mean initial influence
        - avg_final_influence: mean final influence
    """
    df = influence_gain_df.copy()
    
    # Group by player
    agg_dict = {
        'influence_gain_rate': ['mean', 'std', 'count'],
        'initial_influence': 'mean',
        'final_influence': 'mean',
        'influence_change': 'mean',
        'time_elapsed': 'mean'
    }
    
    # Add player info columns if they exist
    group_cols = ['nfl_id']
    for col in ['player_name', 'player_position']:
        if col in df.columns:
            group_cols.append(col)
    
    player_stats = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    player_stats.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col 
        for col in player_stats.columns
    ]
    
    # Rename columns for clarity
    player_stats = player_stats.rename(columns={
        'influence_gain_rate_mean': 'avg_influence_gain_rate',
        'influence_gain_rate_std': 'std_influence_gain_rate',
        'influence_gain_rate_count': 'num_plays',
        'initial_influence_mean': 'avg_initial_influence',
        'final_influence_mean': 'avg_final_influence',
        'influence_change_mean': 'avg_influence_change',
        'time_elapsed_mean': 'avg_time_elapsed'
    })
    
    # Sort by average gain rate
    player_stats = player_stats.sort_values('avg_influence_gain_rate', ascending=False)
    
    return player_stats


# =============================================================================
# Main Pipeline
# =============================================================================

def compute_defender_metrics(
    input_dir: str = 'with_field_control',
    output_dir: str = 'train',
    save_path: Optional[str] = None,
    use_zscore: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline to compute defender field control gain metrics.
    
    Args:
        input_dir: Directory with processed input files
        output_dir: Directory with output files
        save_path: Optional path to save results
        use_zscore: If True, use z-score normalized influence (default True)
        
    Returns:
        Tuple of (play_level_metrics, player_level_metrics)
    """
    print("Step 1: Loading and joining input/output data...")
    combined = load_and_join_input_output(input_dir, output_dir)
    
    if len(combined) == 0:
        print("No data to process!")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"\nStep 2: Computing influence for all frames...")
    print("  (Using last input frame's ball position and velocity for output frames)")
    combined = compute_influence_for_output_frames(combined)
    
    if use_zscore:
        print(f"\nStep 3: Computing z-score normalized influence...")
        combined = compute_zscore_normalized_influence(combined)
    
    print(f"\nStep 4: Calculating influence gain rates...")
    play_metrics = calculate_influence_gain_rate(combined, use_zscore=use_zscore)
    
    print(f"\nStep 5: Aggregating player-level metrics...")
    player_metrics = calculate_player_average_gain_rate(play_metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Normalization: {'Z-score' if use_zscore else 'Raw'}")
    print(f"Plays analyzed: {len(play_metrics)}")
    print(f"Unique players: {len(player_metrics)}")
    
    if len(play_metrics) > 0:
        print(f"\nInfluence Gain Rate (per second):")
        print(f"  Mean: {play_metrics['influence_gain_rate'].mean():.6f}")
        print(f"  Std:  {play_metrics['influence_gain_rate'].std():.6f}")
        print(f"  Min:  {play_metrics['influence_gain_rate'].min():.6f}")
        print(f"  Max:  {play_metrics['influence_gain_rate'].max():.6f}")
    
    # Save if requested
    if save_path:
        play_metrics.to_csv(f"{save_path}_play_level.csv", index=False)
        player_metrics.to_csv(f"{save_path}_player_level.csv", index=False)
        print(f"\nSaved results to {save_path}_play_level.csv and {save_path}_player_level.csv")
    
    return play_metrics, player_metrics


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute defender field control gain metrics')
    parser.add_argument('--input_dir', type=str, default='with_field_control',
                        help='Directory containing processed input files')
    parser.add_argument('--output_dir', type=str, default='train',
                        help='Directory containing output files')
    parser.add_argument('--save', type=str, default='defender_metrics',
                        help='Base path for saving results')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw influence instead of z-score normalized')
    
    args = parser.parse_args()
    
    play_metrics, player_metrics = compute_defender_metrics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_path=args.save,
        use_zscore=not args.raw
    )
    
    print("\nTop 10 players by average influence gain rate:")
    print(player_metrics.head(10))
