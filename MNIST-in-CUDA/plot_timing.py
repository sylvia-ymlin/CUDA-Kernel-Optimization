"""
Timing Breakdown Visualization for MNIST in CUDA
Morandi vintage color palette with clean, spacious design.
"""
import matplotlib.pyplot as plt
import numpy as np

# ============ SciPainter Color Palette ============
COLORS = {
    'bg': '#FAFAFA',           # Clean white background
    'text': '#3D505A',         # Dark gray-blue text
    'grid': '#E0E0E0',         # Light grid
    'blue': '#699ECA',         # Sky blue
    'orange': '#FF8C00',       # Bright orange
    'pink': '#F898CB',         # Candy pink
    'green': '#4DAF4A',        # Fresh green
    'yellow': '#FFCB5B',       # Sunny yellow
    'cyan': '#0098B2',         # Cyan/teal
}

colors = [COLORS['green'], COLORS['blue'], COLORS['pink'], 
          COLORS['yellow'], COLORS['orange'], COLORS['cyan'], '#CCCCCC']  # 7 colors (+ gray for Other)

# Set global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': COLORS['bg'],
    'figure.facecolor': COLORS['bg'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'text.color': COLORS['text'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data (v5/v6 have different timing: async operations)
versions = ['v1 PyTorch', 'v2 NumPy', 'v3 C', 'v4 CUDA', 'v5 cuBLAS', 'v6 Streams']
totals = [3.4, 21.0, 379.7, 1.7, 0.72, 0.47]

# Calculate "Other" to make each version sum to 100%
# Other includes: Python interpreter overhead, CUDA init, epoch loops, print, etc.
raw_data = {
    'Data Loading': [0.06, 0.02, 0.00, 0.13, 0.13, 0.01],
    'Forward': [0.64, 5.42, 269.2, 0.86, 0.00, 0.00],
    'Loss': [0.32, 0.55, 0.00, 0.00, 0.00, 0.00],
    'Backward': [1.51, 9.87, 105.2, 0.44, 0.00, 0.00],
    'Updates': [0.74, 5.15, 3.04, 0.17, 0.00, 0.00],
    'GPU Compute': [0.00, 0.00, 0.00, 0.00, 0.59, 0.46],
}

# Calculate Other = total - sum(all categories)
other = []
for i, t in enumerate(totals):
    measured = sum(raw_data[k][i] for k in raw_data)
    other.append(max(0, t - measured))  # Ensure non-negative

data = raw_data.copy()
data['Other'] = other  # Python/CUDA overhead, epoch loops, print, etc.

# ============ Figure 1: Percentage Breakdown (Horizontal) ============
fig, ax = plt.subplots(figsize=(12, 6))

# Convert to percentages
percentages = {k: [v / t * 100 for v, t in zip(vals, totals)] for k, vals in data.items()}

y = np.arange(len(versions))
height = 0.6
left = np.zeros(len(versions))

for i, (label, values) in enumerate(percentages.items()):
    bars = ax.barh(y, values, height, left=left, label=label, color=colors[i], edgecolor='white', linewidth=0.5)
    left += np.array(values)

# Add total time labels on left (before version name)
for i, (v, t) in enumerate(zip(versions, totals)):
    # Format time nicely
    if t >= 100:
        time_str = f'{t:.0f}s'
    elif t >= 1:
        time_str = f'{t:.1f}s'
    else:
        time_str = f'{t:.2f}s'
    ax.text(103, i, time_str, va='center', ha='left', fontsize=11, fontweight='bold', color=COLORS['text'])

ax.set_xlim(0, 120)
ax.set_xlabel('Time Distribution (%)', fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels(versions, fontsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=7, frameon=False, fontsize=9)
ax.set_title('MNIST MLP Training · Time Breakdown (% of total)', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, color=COLORS['grid'])

# Add percentage markers
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

plt.tight_layout()
plt.savefig('assets/timing_analysis.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("Saved: timing_analysis.png")

# ============ Figure 2: Version Progression Flowchart ============
fig2, ax2 = plt.subplots(figsize=(16, 6))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Version data
flow_data = [
    {'name': 'v1.py', 'tech': 'PyTorch', 'time': '3.4s', 'speedup': '112×', 'color': COLORS['blue']},
    {'name': 'v2.py', 'tech': 'NumPy', 'time': '21.0s', 'speedup': '18×', 'color': COLORS['green']},
    {'name': 'v3.c', 'tech': 'C CPU', 'time': '379.7s', 'speedup': '1× (base)', 'color': COLORS['orange']},
    {'name': 'v4.cu', 'tech': 'CUDA', 'time': '1.7s', 'speedup': '223×', 'color': COLORS['pink']},
    {'name': 'v5.cu', 'tech': 'cuBLAS', 'time': '0.72s', 'speedup': '527×', 'color': COLORS['yellow']},
    {'name': 'v6.cu', 'tech': 'Streams', 'time': '0.47s', 'speedup': '808×', 'color': COLORS['cyan']},
]

# Box positions
x_positions = [1.2, 3.2, 5.2, 7.2, 9.2, 11.2]
y_center = 3.0
box_w, box_h = 1.5, 2.2

# Draw boxes and content
for i, (x, d) in enumerate(zip(x_positions, flow_data)):
    # Box
    rect = plt.Rectangle((x - box_w/2, y_center - box_h/2), box_w, box_h,
                          facecolor=d['color'], edgecolor='white', linewidth=2, 
                          alpha=0.85, zorder=2)
    ax2.add_patch(rect)
    
    # Version name
    ax2.text(x, y_center + 0.6, d['name'], ha='center', va='center', 
             fontsize=13, fontweight='bold', color='white', zorder=3)
    # Tech
    ax2.text(x, y_center + 0.15, d['tech'], ha='center', va='center', 
             fontsize=11, color='white', alpha=0.9, zorder=3)
    # Time
    ax2.text(x, y_center - 0.35, d['time'], ha='center', va='center', 
             fontsize=14, fontweight='bold', color='white', zorder=3)
    # Speedup
    ax2.text(x, y_center - 0.8, d['speedup'], ha='center', va='center', 
             fontsize=10, color='white', alpha=0.85, zorder=3)

# Draw arrows between boxes
arrow_style = dict(arrowstyle='->', color=COLORS['text'], lw=2, mutation_scale=15)
for i in range(len(x_positions) - 1):
    x_start = x_positions[i] + box_w/2 + 0.05
    x_end = x_positions[i+1] - box_w/2 - 0.05
    ax2.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                 arrowprops=arrow_style, zorder=1)

# Labels below arrows
transitions = [
    ('Remove\nautograd', 1),
    ('Remove\nBLAS', 2),
    ('GPU\nparallel', 3),
    ('Use\ncuBLAS', 4),
    ('Async\nStreams', 5),
]
for label, idx in transitions:
    x_mid = (x_positions[idx-1] + x_positions[idx]) / 2
    ax2.text(x_mid, y_center - 1.6, label, ha='center', va='top', 
             fontsize=9, color=COLORS['text'], alpha=0.8)

# Title
ax2.text(7, 5.4, 'Version Progression · MNIST MLP Training', 
         ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])

# Subtitle
ax2.text(7, 4.9, '784 → 1024 → 10  |  10 epochs  |  batch 32', 
         ha='center', va='center', fontsize=10, color=COLORS['text'], alpha=0.7)

plt.tight_layout()
plt.savefig('assets/speedup_comparison.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("Saved: speedup_comparison.png")

