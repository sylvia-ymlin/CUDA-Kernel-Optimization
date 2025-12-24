"""
Timing Breakdown Visualization for MNIST in CUDA
Morandi vintage color palette with clean, spacious design.
"""
import matplotlib.pyplot as plt
import numpy as np

# ============ Morandi Color Palette ============
MORANDI = {
    'bg': '#F5F1EB',           # Warm cream background
    'text': '#5D5D5D',         # Soft gray text
    'grid': '#D4CFC7',         # Muted grid
    'sage': '#9CAF88',         # Sage green
    'dusty_blue': '#8BA7B9',   # Dusty blue
    'terracotta': '#C4A484',   # Terracotta/tan
    'mauve': '#B8A9C9',        # Soft mauve
    'blush': '#D4A5A5',        # Dusty rose
}

colors = [MORANDI['sage'], MORANDI['dusty_blue'], MORANDI['blush'], 
          MORANDI['mauve'], MORANDI['terracotta'], '#A89B8C']  # 6 colors for 6 categories

# Set global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.facecolor': MORANDI['bg'],
    'figure.facecolor': MORANDI['bg'],
    'axes.edgecolor': MORANDI['grid'],
    'axes.labelcolor': MORANDI['text'],
    'xtick.color': MORANDI['text'],
    'ytick.color': MORANDI['text'],
    'text.color': MORANDI['text'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data (v5 has different timing: H2D + GPU compute only)
versions = ['v1 PyTorch', 'v2 NumPy', 'v3 C', 'v4 CUDA', 'v5 cuBLAS']
totals = [3.4, 21.0, 379.7, 1.7, 0.72]

# For v5: Data Loading = H2D (0.13s), GPU Compute (0.59s) combines Forward+Loss+Backward+Updates
data = {
    'Data Loading': [0.06, 0.02, 0.00, 0.13, 0.13],
    'Forward': [0.64, 5.42, 269.2, 0.86, 0.00],
    'Loss': [0.32, 0.55, 0.00, 0.00, 0.00],
    'Backward': [1.51, 9.87, 105.2, 0.44, 0.00],
    'Updates': [0.74, 5.15, 3.04, 0.17, 0.00],
    'GPU Compute': [0.00, 0.00, 0.00, 0.00, 0.59],  # v5 only (cuBLAS combined)
}

# ============ Figure 1: Percentage Breakdown (Horizontal) ============
fig, ax = plt.subplots(figsize=(10, 5))

# Convert to percentages
percentages = {k: [v / t * 100 for v, t in zip(vals, totals)] for k, vals in data.items()}

y = np.arange(len(versions))
height = 0.55
left = np.zeros(len(versions))

for i, (label, values) in enumerate(percentages.items()):
    bars = ax.barh(y, values, height, left=left, label=label, color=colors[i], edgecolor='white', linewidth=0.5)
    left += np.array(values)

# Add time labels on right
for i, (v, t) in enumerate(zip(versions, totals)):
    ax.text(102, i, f'{t}s', va='center', ha='left', fontsize=10, fontweight='bold', color=MORANDI['text'])

ax.set_xlim(0, 115)
ax.set_xlabel('Time Distribution (%)', fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels(versions, fontsize=11)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6, frameon=False, fontsize=9)
ax.set_title('MNIST MLP Training · Time Breakdown', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, color=MORANDI['grid'])

plt.tight_layout()
plt.savefig('assets/timing_analysis.png', dpi=150, bbox_inches='tight', facecolor=MORANDI['bg'])
print("Saved: timing_analysis.png")

# ============ Figure 2: Version Progression Flowchart ============
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Version data
flow_data = [
    {'name': 'v1.py', 'tech': 'PyTorch', 'time': '3.4s', 'speedup': '112×', 'color': MORANDI['dusty_blue']},
    {'name': 'v2.py', 'tech': 'NumPy', 'time': '21.0s', 'speedup': '18×', 'color': MORANDI['sage']},
    {'name': 'v3.c', 'tech': 'C CPU', 'time': '379.7s', 'speedup': '1× (base)', 'color': MORANDI['terracotta']},
    {'name': 'v4.cu', 'tech': 'CUDA', 'time': '1.7s', 'speedup': '223×', 'color': MORANDI['blush']},
    {'name': 'v5.cu', 'tech': 'cuBLAS', 'time': '0.72s', 'speedup': '527×', 'color': MORANDI['mauve']},
]

# Box positions
x_positions = [1.2, 3.4, 5.6, 7.8, 10.0]
y_center = 3.0
box_w, box_h = 1.6, 2.2

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
arrow_style = dict(arrowstyle='->', color=MORANDI['text'], lw=2, mutation_scale=15)
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
]
for label, idx in transitions:
    x_mid = (x_positions[idx-1] + x_positions[idx]) / 2
    ax2.text(x_mid, y_center - 1.6, label, ha='center', va='top', 
             fontsize=9, color=MORANDI['text'], alpha=0.8)

# Title
ax2.text(6, 5.4, 'Version Progression · MNIST MLP Training', 
         ha='center', va='center', fontsize=14, fontweight='bold', color=MORANDI['text'])

# Subtitle
ax2.text(6, 4.9, '784 → 1024 → 10  |  10 epochs  |  batch 32', 
         ha='center', va='center', fontsize=10, color=MORANDI['text'], alpha=0.7)

plt.tight_layout()
plt.savefig('assets/speedup_comparison.png', dpi=150, bbox_inches='tight', facecolor=MORANDI['bg'])
print("Saved: speedup_comparison.png")

