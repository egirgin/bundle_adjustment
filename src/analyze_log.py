import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_log_file(filepath='logfile.txt'):
    """Parses the visual odometry log file to extract key metrics, including the keyframe trigger reason."""
    data = []
    current_frame_data = {}
    is_first_keyframe = True
    gba_improvement = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if 'Processing frame' in line:
                if current_frame_data:
                    data.append(current_frame_data)
                m = re.search(r'Processing frame (\d+)...', line)
                if m:
                    current_frame_data = {'frame_id': int(m.group(1))}

            elif 'Pose Estimation' in line and current_frame_data:
                m = re.search(r'Ratio: ([\d.]+)', line)
                if m:
                    current_frame_data['inlier_ratio'] = float(m.group(1))

            # --- MODIFICATION: Capture the trigger reason ---
            elif 'Keyframe Trigger' in line and current_frame_data:
                if is_first_keyframe:
                    # Manually add the first keyframe (frame 0), which is just for initialization
                    data.append({'frame_id': 0, 'is_keyframe': True, 'kf_trigger_reason': 'Initialization'})
                    is_first_keyframe = False
                
                current_frame_data['is_keyframe'] = True
                m_reason = re.search(r'Keyframe Trigger: ([\w\s]+) \(', line)
                if m_reason:
                    current_frame_data['kf_trigger_reason'] = m_reason.group(1).strip()
            # --- END MODIFICATION ---

            elif 'LBA Complete' in line and 'Improvement' in line and current_frame_data:
                m = re.search(r'Improvement: ([-\d.]+)%', line)
                if m:
                    current_frame_data['lba_improvement'] = float(m.group(1))

            elif 'Global Bundle Adjustment' in line and 'Improvement' in line:
                m = re.search(r'Improvement: ([-\d.]+)%', line)
                if m:
                    gba_improvement = float(m.group(1))

    if current_frame_data:
        data.append(current_frame_data)

    return pd.DataFrame(data), gba_improvement

def analyze_and_plot(df, gba_improvement):
    """Generates plots and analysis, color-coding keyframes by their trigger criteria."""
    if df.empty:
        print("Could not parse any data from the log file.")
        return

    kf_df = df[df['is_keyframe'] == True].copy()
    kf_df['keyframe_index'] = range(len(kf_df))

    print("--- Log File Analysis ---")
    print(f"Total Frames Processed: {df['frame_id'].max() + 1}")
    print(f"Total Keyframes Created: {len(kf_df)}")
    if gba_improvement is not None:
        print(f"Final Global BA Improvement: {gba_improvement:.2f}%")

    # --- MODIFICATION: Plotting logic for trigger reasons ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=False)
    fig.suptitle('Visual Odometry Pipeline Performance Analysis', fontsize=16)

    # 1. Inlier Ratio at the moment of Keyframe creation, colored by trigger
    axes[0].axhline(y=0.4, color='orange', linestyle='--', label='Acceptable Threshold (0.40)')
    axes[0].axhline(y=0.2, color='r', linestyle='--', label='Risky Threshold (0.20)')

    trigger_colors = {
        'Pixel Displacement': 'blue',
        'Rotation': 'green',
        'Feature Ratio': 'red',
        'Initialization': 'purple'
    }
    
    # Create a list of colors for each point in the scatter plot
    colors = kf_df['kf_trigger_reason'].map(trigger_colors).fillna('gray')
    
    # Plot all keyframe points
    axes[0].scatter(kf_df['keyframe_index'], kf_df['inlier_ratio'], c=colors, s=70, zorder=5, alpha=0.8)

    # Create a custom legend for the colors
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=reason,
                              markerfacecolor=color, markersize=12)
                       for reason, color in trigger_colors.items()]
    axes[0].legend(handles=legend_elements, title="Keyframe Trigger Reason")
    
    axes[0].set_ylabel('Inlier Ratio')
    axes[0].set_title('Front-End Quality When Creating Keyframes')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(-1, len(kf_df))


    # 2. LBA Improvement per Keyframe
    lba_data = kf_df.dropna(subset=['lba_improvement'])
    bar_colors = ['red' if x < 0 else 'teal' for x in lba_data['lba_improvement']]
    axes[1].bar(lba_data['keyframe_index'], lba_data['lba_improvement'], color=bar_colors)
    axes[1].axhline(y=0, color='black', linestyle='-')
    axes[1].set_ylabel('Cost Improvement (%)')
    axes[1].set_title('Local Bundle Adjustment (LBA) Performance per Keyframe')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Keyframe Number')
    axes[1].set_xlim(-1, len(kf_df))

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("vo_analysis_plot_by_trigger.png")
    print("\nGenerated analysis plot: vo_analysis_plot_by_trigger.png")
    plt.show()


if __name__ == '__main__':
    # Make sure to have your logfile.txt in the same directory
    log_df, gba_imp = parse_log_file()
    analyze_and_plot(log_df, gba_imp)