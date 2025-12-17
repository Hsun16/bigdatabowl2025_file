"""
Pre-Throw Defender Field Control Gain Metrics

This module computes metrics to measure how quickly defenders close down space
BEFORE the ball is thrown, based on their gain of field influence at the ball
landing point using only input data.

Key Functions:
- load_input_data: Load input tracking data for defenders
- calculate_influence_gain_rate: Compute influence gain per player (reused)
- compute_pre_throw_defender_metrics: Full pipeline for pre-throw metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional

# Import from defender_metrics.py (same directory)
from defender_metrics import (
    compute_influence_direct,
    compute_zscore_normalized_influence,
    calculate_player_average_gain_rate
)

# Import from field_control.py
from field_control import influence_radius_vec


# =============================================================================
# Data Loading
# =============================================================================

def load_input_data(
    input_dir: str = 'with_field_control',
    input_pattern: str = 'input_2023_w*.csv',
    filter_defenders: bool = True
) -> pd.DataFrame:
    """
    Load input tracking data for defenders.
    
    Args:
        input_dir: Directory containing processed input files (with field control)
        input_pattern: Glob pattern for input files
        filter_defenders: If True, only include defensive players
        
    Returns:
        DataFrame with input tracking data
    """
    input_path = Path(input_dir)
    
    # Find input files
    input_files = sorted(input_path.glob(input_pattern))
    
    print(f"Found {len(input_files)} input files")
    
    all_data = []
    
    for input_file in tqdm(input_files, desc="Loading input files"):
        # Load data
        df = pd.read_csv(input_file)
        
        # Filter to defenders if requested
        if filter_defenders:
            if 'player_side' in df.columns:
                df = df[df['player_side'].str.lower() == 'defense'].copy()
            else:
                print(f"  Warning: No player_side column in {input_file.name}")
                continue
        
        if len(df) > 0:
            all_data.append(df)
    
    if len(all_data) == 0:
        print("No data found!")
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"Loaded data: {len(result)} rows")
    
    return result


# =============================================================================
# Influence Gain Rate Calculation (Pre-Throw)
# =============================================================================

def calculate_pre_throw_influence_gain_rate(
    input_df: pd.DataFrame,
    fps: float = 10.0,
    use_zscore: bool = True
) -> pd.DataFrame:
    """
    Calculate influence gain rate BEFORE the ball is thrown.
    
    For each player in each play:
    - Get influence at first frame
    - Get influence at last frame (just before throw)
    - Compute: (final_influence - initial_influence) / (num_frames / fps)
    
    Args:
        input_df: DataFrame with input tracking data (pre-throw only)
        fps: Frames per second (default 10)
        use_zscore: If True, use z-score normalized influence (default True)
        
    Returns:
        DataFrame with per-player influence gain metrics
    """
    df = input_df.copy()
    
    # Compute influence if not already present
    if 'influence_at_land' not in df.columns:
        if 'vx' not in df.columns or 'vy' not in df.columns:
            raise ValueError("DataFrame must have 'vx' and 'vy' columns")
        if 'ball_land_x' not in df.columns or 'ball_land_y' not in df.columns:
            raise ValueError("DataFrame must have 'ball_land_x' and 'ball_land_y' columns")
        df = compute_influence_direct(df, 'ball_land_x', 'ball_land_y')
    
    # Compute z-score normalized influence if requested
    if use_zscore:
        if 'influence_zscore' not in df.columns:
            df = compute_zscore_normalized_influence(df)
        influence_col = 'influence_zscore'
    else:
        influence_col = 'influence_at_land'
    
    results = []
    
    # Group by player in each play
    for (game_id, play_id, nfl_id), player_data in df.groupby(['game_id', 'play_id', 'nfl_id']):
        
        # Sort by frame
        player_data = player_data.sort_values('frame_id')
        
        if len(player_data) < 2:
            continue
        
        # Get initial influence (first frame)
        initial_influence = player_data[influence_col].iloc[0]
        
        # Get final influence (last frame = just before throw)
        final_influence = player_data[influence_col].iloc[-1]
        
        # Skip if either is NaN
        if np.isnan(initial_influence) or np.isnan(final_influence):
            continue
        
        # Number of frames
        num_frames = len(player_data)
        
        # Time elapsed in seconds (num_frames - 1 intervals)
        time_elapsed = (num_frames - 1) / fps
        
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
            'num_frames': num_frames,
            'time_elapsed': time_elapsed,
            'influence_gain_rate': influence_gain_rate,
            'normalized_method': 'zscore' if use_zscore else 'raw',
            'phase': 'pre_throw'
        }
        
        # Add optional player info
        for col in ['player_name', 'player_position', 'player_role', 'player_side']:
            if col in player_data.columns:
                val = player_data[col].dropna().iloc[0] if len(player_data[col].dropna()) > 0 else None
                player_info[col] = val
        
        results.append(player_info)
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        # Sort by influence gain rate
        result_df = result_df.sort_values('influence_gain_rate', ascending=False)
    
    return result_df


# =============================================================================
# Main Pipeline
# =============================================================================

def compute_pre_throw_defender_metrics(
    input_dir: str = 'with_field_control',
    save_path: Optional[str] = None,
    use_zscore: bool = True,
    min_frames: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline to compute pre-throw defender field control gain metrics.
    
    Args:
        input_dir: Directory with processed input files
        save_path: Optional path to save results
        use_zscore: If True, use z-score normalized influence (default True)
        min_frames: Minimum number of frames required per play (default 5)
        
    Returns:
        Tuple of (play_level_metrics, player_level_metrics)
    """
    print("Step 1: Loading input data (defenders only)...")
    input_data = load_input_data(input_dir)
    
    if len(input_data) == 0:
        print("No data to process!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter plays with minimum frames
    print(f"\nStep 2: Filtering plays with at least {min_frames} frames...")
    frame_counts = input_data.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].count()
    valid_keys = frame_counts[frame_counts >= min_frames].reset_index()[['game_id', 'play_id', 'nfl_id']]
    input_data = input_data.merge(valid_keys, on=['game_id', 'play_id', 'nfl_id'], how='inner')
    print(f"  Remaining data: {len(input_data)} rows")
    
    print(f"\nStep 3: Computing influence...")
    if 'influence_at_land' not in input_data.columns:
        input_data = compute_influence_direct(input_data, 'ball_land_x', 'ball_land_y')
    
    if use_zscore:
        print(f"\nStep 4: Computing z-score normalized influence...")
        input_data = compute_zscore_normalized_influence(input_data)
    
    print(f"\nStep 5: Calculating pre-throw influence gain rates...")
    play_metrics = calculate_pre_throw_influence_gain_rate(input_data, use_zscore=use_zscore)
    
    print(f"\nStep 6: Aggregating player-level metrics...")
    player_metrics = calculate_player_average_gain_rate(play_metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("PRE-THROW RESULTS SUMMARY")
    print("="*60)
    print(f"Normalization: {'Z-score' if use_zscore else 'Raw'}")
    print(f"Plays analyzed: {len(play_metrics)}")
    print(f"Unique players: {len(player_metrics)}")
    
    if len(play_metrics) > 0:
        print(f"\nPre-Throw Influence Gain Rate (per second):")
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
    
    parser = argparse.ArgumentParser(description='Compute pre-throw defender field control gain metrics')
    parser.add_argument('--input_dir', type=str, default='with_field_control',
                        help='Directory containing processed input files')
    parser.add_argument('--save', type=str, default='pre_throw_defender_metrics',
                        help='Base path for saving results')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw influence instead of z-score normalized')
    parser.add_argument('--min_frames', type=int, default=5,
                        help='Minimum number of frames required per play')
    
    args = parser.parse_args()
    
    play_metrics, player_metrics = compute_pre_throw_defender_metrics(
        input_dir=args.input_dir,
        save_path=args.save,
        use_zscore=not args.raw,
        min_frames=args.min_frames
    )
    
    print("\nTop 10 players by average pre-throw influence gain rate:")
    print(player_metrics.head(10))
