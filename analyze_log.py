import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(filepath='logfile.txt'):
    """Parses the visual odometry log file to extract key metrics."""
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

            elif 'Keyframe Trigger' in line and current_frame_data:
                if is_first_keyframe:
                    data.append({'frame_id': 0, 'is_keyframe': True})
                    is_first_keyframe = False
                current_frame_data['is_keyframe'] = True
            
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
    """Generates plots and analysis from the parsed log data."""
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

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Visual Odometry Back-End Performance', fontsize=16)

    # 1. Inlier Ratio at the moment of Keyframe creation
    axes[0].plot(kf_df['keyframe_index'], kf_df['inlier_ratio'], 'bo-', label='Inlier Ratio at KF Creation')
    axes[0].axhline(y=0.2, color='r', linestyle='--', label='Reliability Threshold (0.20)')
    axes[0].set_ylabel('Inlier Ratio')
    axes[0].set_title('Front-End Quality When Creating Keyframes')
    axes[0].grid(True, linestyle='--', linewidth=0.5)
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # 2. LBA Improvement per Keyframe
    lba_data = kf_df.dropna(subset=['lba_improvement'])
    colors = ['red' if x < 0 else 'teal' for x in lba_data['lba_improvement']]
    axes[1].bar(lba_data['keyframe_index'], lba_data['lba_improvement'], color=colors)
    axes[1].axhline(y=0, color='black', linestyle='-')
    axes[1].set_ylabel('Cost Improvement (%)')
    axes[1].set_title('Local Bundle Adjustment (LBA) Performance per Keyframe')
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Keyframe Number')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("vo_analysis_plot_by_keyframe.png")
    print("\nGenerated analysis plot: vo_analysis_plot_by_keyframe.png")
    plt.show()

if __name__ == '__main__':
    log_df, gba_imp = parse_log_file()
    analyze_and_plot(log_df, gba_imp)