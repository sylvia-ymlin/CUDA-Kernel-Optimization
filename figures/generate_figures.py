#!/usr/bin/env python3
"""
Generate professional figures for CUDA Kernel Optimization README.

Run: python figures/generate_figures.py

Dependencies: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Set style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# FIGURE 1: Memory Hierarchy
# ============================================================================
def generate_memory_hierarchy():
    """Generate GPU memory hierarchy diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'registers': '#2ecc71',    # Green
        'shared': '#3498db',       # Blue
        'l2': '#9b59b6',           # Purple
        'global': '#e74c3c',       # Red
        'arrow': '#34495e'         # Dark gray
    }
    
    # Layer definitions: (y_center, height, width, label, sublabel, bandwidth, color)
    layers = [
        (8.5, 1.0, 3.0, 'Registers', '255 KB/SM', '~8 TB/s', colors['registers']),
        (6.5, 1.2, 5.0, 'Shared Memory / L1', '48 KB/block', '~12 TB/s', colors['shared']),
        (4.3, 1.2, 7.0, 'L2 Cache', '4 MB', '~3 TB/s', colors['l2']),
        (2.0, 1.5, 9.0, 'Global Memory (HBM/GDDR)', '15 GB', '320 GB/s', colors['global']),
    ]
    
    for y, h, w, label, size, bw, color in layers:
        # Draw rounded rectangle
        rect = FancyBboxPatch((5 - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=color, edgecolor='black', linewidth=2,
                              alpha=0.85)
        ax.add_patch(rect)
        
        # Add label
        ax.text(5, y + 0.1, label, ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')
        ax.text(5, y - 0.25, f'{size}  |  {bw}', ha='center', va='center',
                fontsize=10, color='white')
    
    # Add arrows showing data flow
    arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2,
                       connectionstyle='arc3,rad=0')
    
    for i in range(len(layers) - 1):
        y_top = layers[i][0] - layers[i][1]/2 - 0.1
        y_bottom = layers[i+1][0] + layers[i+1][1]/2 + 0.1
        ax.annotate('', xy=(5, y_bottom), xytext=(5, y_top),
                    arrowprops=arrow_style)
    
    # Add latency annotations on the right
    latencies = [
        (8.5, '1 cycle'),
        (6.5, '~20 cycles'),
        (4.3, '~200 cycles'),
        (2.0, '~400-600 cycles'),
    ]
    
    for y, lat in latencies:
        ax.text(9.5, y, lat, ha='right', va='center', fontsize=10,
                color='#555555', style='italic')
    
    # Title and annotations
    ax.text(5, 9.5, 'GPU Memory Hierarchy (Tesla T4)', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(0.5, 0.5, 'Capacity increases ↓  |  Bandwidth decreases ↓  |  Latency increases ↓',
            ha='left', va='center', fontsize=10, color='#666666')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_hierarchy.png'), bbox_inches='tight')
    plt.close()
    print("✓ Generated: memory_hierarchy.png")


# ============================================================================
# FIGURE 2: Roofline Model (Log-Log Plot)
# ============================================================================
def generate_roofline():
    """Generate roofline model plot with kernel placements."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Tesla T4 specs
    peak_gflops = 8141
    peak_bandwidth_gb = 320
    ridge_point = peak_gflops / peak_bandwidth_gb  # ~25.4 FLOP/byte
    
    # Arithmetic intensity range (log scale)
    ai = np.logspace(-1.5, 2.5, 500)
    
    # Roofline: min(peak_compute, peak_bandwidth * AI)
    memory_bound = peak_bandwidth_gb * ai
    roofline = np.minimum(peak_gflops, memory_bound)
    
    # Plot roofline
    ax.loglog(ai, roofline, 'b-', linewidth=3, label='Roofline (Tesla T4)')
    
    # Fill regions
    ax.fill_between(ai, roofline, 0.1, where=(ai < ridge_point),
                    alpha=0.15, color='blue', label='Memory-bound region')
    ax.fill_between(ai, roofline, 0.1, where=(ai >= ridge_point),
                    alpha=0.15, color='red', label='Compute-bound region')
    
    # Ridge point
    ax.axvline(x=ridge_point, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(ridge_point * 1.1, 50, f'Ridge Point\n({ridge_point:.1f} FLOP/byte)',
            fontsize=9, color='gray', va='bottom')
    
    # Peak lines
    ax.axhline(y=peak_gflops, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(0.015, peak_gflops * 1.08, f'Peak Compute: {peak_gflops} GFLOPS',
            fontsize=9, color='red', ha='left', va='bottom')
    
    # Kernel data points with custom label positions (x_offset, y_offset in points)
    kernels = [
        # (name, AI, achieved_gflops, color, marker, x_off, y_off, ha)
        ('Elementwise (79% BW)', 0.125, 0.125 * 252, '#2ecc71', 'o', 12, 0, 'left'),
        ('Reduction (91% BW)', 0.125, 0.125 * 290, '#27ae60', 's', 12, 5, 'left'),
        ('Transpose (62% BW)', 0.0625, 0.0625 * 199, '#3498db', '^', 12, -5, 'left'),
        ('SGEMM v7 (65% cuBLAS)', 170, 4209, '#e74c3c', 'D', 15, -15, 'left'),
        ('cuBLAS (reference)', 170, 6523, '#9b59b6', '*', 15, 10, 'left'),
    ]
    
    for name, ai_val, gflops, color, marker, x_off, y_off, ha in kernels:
        ax.scatter([ai_val], [gflops], c=color, s=150, marker=marker, 
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(name, (ai_val, gflops), xytext=(x_off, y_off),
                   textcoords='offset points', fontsize=9,
                   fontweight='bold', color=color, ha=ha,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='none', alpha=0.7))
    
    # Realistic ceilings (dashed lines)
    practical_bw = 280  # After ECC, controller overhead
    practical_memory = practical_bw * ai[ai < ridge_point]
    ax.loglog(ai[ai < ridge_point], practical_memory, 'b--', 
              linewidth=1.5, alpha=0.5, label=f'Practical BW ({practical_bw} GB/s)')
    
    # Labels and formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('Roofline Model: Tesla T4 (FP32)', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(1, 12000)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roofline_model.png'), bbox_inches='tight')
    plt.close()
    print("✓ Generated: roofline_model.png")


# ============================================================================
# FIGURE 3: SGEMM Tiling Visualization
# ============================================================================
def generate_sgemm_tiling():
    """Generate SGEMM 2D tiling visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors
    colors = {
        'A': '#3498db',
        'B': '#2ecc71', 
        'C': '#e74c3c',
        'tile': '#f39c12',
        'thread': '#9b59b6',
        'highlight': '#ffeb3b'
    }
    
    # --- Panel 1: Block-level tiling ---
    ax = axes[0]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_title('Block-Level Tiling\n(Shared Memory)', fontsize=12, fontweight='bold')
    
    # Matrix C (output)
    for i in range(8):
        for j in range(8):
            color = colors['C'] if (i // 4 == 0 and j // 4 == 0) else '#ffcccc'
            alpha = 0.8 if (i // 4 == 0 and j // 4 == 0) else 0.3
            rect = plt.Rectangle((j, 7-i), 1, 1, facecolor=color, 
                                  edgecolor='black', alpha=alpha, linewidth=0.5)
            ax.add_patch(rect)
    
    # Highlight one block
    rect = plt.Rectangle((0, 4), 4, 4, facecolor='none',
                          edgecolor='black', linewidth=3)
    ax.add_patch(rect)
    
    ax.text(2, 6, 'Block\n(BM×BN)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(4, -0.3, 'Matrix C (M×N)', ha='center', fontsize=10)
    ax.axis('off')
    
    # --- Panel 2: Thread-level tiling (2D) ---
    ax = axes[1]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_title('Thread-Level Tiling (2D)\n(Registers)', fontsize=12, fontweight='bold')
    
    # Block divided into thread tiles
    thread_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
                     '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    
    for ti in range(4):
        for tj in range(4):
            color = thread_colors[(ti * 4 + tj) % 8]
            rect = plt.Rectangle((tj * 2, (3-ti) * 2), 2, 2, 
                                  facecolor=color, edgecolor='black', 
                                  alpha=0.6, linewidth=1)
            ax.add_patch(rect)
    
    # Highlight one thread's work
    rect = plt.Rectangle((0, 6), 2, 2, facecolor='none',
                          edgecolor='yellow', linewidth=4)
    ax.add_patch(rect)
    ax.text(1, 7, 'Thread 0\n(TM×TN)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    
    ax.text(4, -0.3, 'One Block (BM×BN)', ha='center', fontsize=10)
    ax.axis('off')
    
    # --- Panel 3: Register usage detail ---
    ax = axes[2]
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_title('Register Caching\n(One Thread)', fontsize=12, fontweight='bold')
    
    # A fragment (column)
    for i in range(8):
        rect = plt.Rectangle((0, 7-i), 1, 1, facecolor=colors['A'],
                              edgecolor='black', alpha=0.7, linewidth=1)
        ax.add_patch(rect)
    ax.text(0.5, -0.3, 'a_frag\n(TM)', ha='center', fontsize=9)
    
    # B fragment (row)
    for j in range(8):
        rect = plt.Rectangle((2+j, 7), 1, 1, facecolor=colors['B'],
                              edgecolor='black', alpha=0.7, linewidth=1)
        ax.add_patch(rect)
    ax.text(6, 8.3, 'b_frag (TN)', ha='center', fontsize=9)
    
    # Accumulator (TM x TN)
    for i in range(8):
        for j in range(8):
            rect = plt.Rectangle((2+j, 5.5-i-0.7*8), 0.9, 0.9, facecolor=colors['C'],
                                  edgecolor='black', alpha=0.5, linewidth=0.5)
            ax.add_patch(rect)
    ax.text(6, -0.8, 'accum[TM][TN]\n(Registers)', ha='center', fontsize=9)
    
    # Arrows showing outer product
    ax.annotate('', xy=(1.8, 3.5), xytext=(1.2, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5, 6.8), xytext=(5, 7.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(5.5, 4.5, '+=\na × b', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sgemm_tiling.png'), bbox_inches='tight')
    plt.close()
    print("✓ Generated: sgemm_tiling.png")


# ============================================================================
# FIGURE 4: Transpose Bank Conflict Visualization
# ============================================================================
def generate_transpose_bank_conflicts():
    """Generate visualization of shared memory bank conflicts in transpose."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Panel 1: With bank conflicts ---
    ax = axes[0]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_title('32×32 Tile (Bank Conflicts)\nColumn read = 32-way conflict', 
                 fontsize=11, fontweight='bold', color='#e74c3c')
    
    # Draw tile with bank coloring
    for i in range(8):
        for j in range(8):
            bank = j % 4  # Simplified: 4 banks shown
            colors_bank = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            rect = plt.Rectangle((j, 7-i), 1, 1, facecolor=colors_bank[bank],
                                  edgecolor='black', alpha=0.6, linewidth=0.5)
            ax.add_patch(rect)
    
    # Highlight column access
    for i in range(8):
        rect = plt.Rectangle((2, 7-i), 1, 1, facecolor='none',
                              edgecolor='black', linewidth=3)
        ax.add_patch(rect)
    
    ax.text(2.5, -0.3, '← All same bank!', ha='center', fontsize=10, 
            fontweight='bold', color='#e74c3c')
    ax.axis('off')
    
    # --- Panel 2: With padding (no conflicts) ---
    ax = axes[1]
    ax.set_xlim(-0.5, 9.5)  # Extra column for padding
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_title('32×33 Tile (With Padding)\nColumn read = No conflict', 
                 fontsize=11, fontweight='bold', color='#2ecc71')
    
    # Draw tile with shifted bank coloring (due to padding)
    for i in range(8):
        for j in range(8):
            # With 33-wide rows, each row shifts bank by 1
            bank = (j + i) % 4
            colors_bank = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            rect = plt.Rectangle((j, 7-i), 1, 1, facecolor=colors_bank[bank],
                                  edgecolor='black', alpha=0.6, linewidth=0.5)
            ax.add_patch(rect)
    
    # Padding column
    for i in range(8):
        rect = plt.Rectangle((8, 7-i), 1, 1, facecolor='#cccccc',
                              edgecolor='black', alpha=0.3, linewidth=0.5)
        ax.add_patch(rect)
    ax.text(8.5, 3.5, 'PAD', ha='center', va='center', fontsize=8, 
            rotation=90, color='#666666')
    
    # Highlight column access
    for i in range(8):
        rect = plt.Rectangle((2, 7-i), 1, 1, facecolor='none',
                              edgecolor='black', linewidth=3)
        ax.add_patch(rect)
    
    ax.text(2.5, -0.3, '← Different banks!', ha='center', fontsize=10,
            fontweight='bold', color='#2ecc71')
    ax.axis('off')
    
    # Legend
    fig.text(0.5, 0.02, 'Colors represent different memory banks. ' +
             'Padding ensures column accesses map to different banks.',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'bank_conflicts.png'), bbox_inches='tight')
    plt.close()
    print("✓ Generated: bank_conflicts.png")


# ============================================================================
# FIGURE 5: Occupancy vs ILP Tradeoff
# ============================================================================
def generate_occupancy_ilp():
    """Generate occupancy vs ILP tradeoff visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points (conceptual)
    versions = ['v2\n(High Occ)', 'v3', 'v4', 'v5', 'v6\n(High ILP)', 'v7']
    occupancy = [100, 75, 62.5, 50, 25, 25]  # Relative %
    ilp = [1, 2, 4, 4, 8, 8]  # Work per thread
    gflops = [643, 1109, 1408, 1477, 4052, 4209]
    
    x = np.arange(len(versions))
    width = 0.35
    
    # Create bars
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, occupancy, width, label='Occupancy (%)', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, gflops, width, label='GFLOPS', 
                    color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add ILP markers
    for i, (v, ilp_val) in enumerate(zip(versions, ilp)):
        ax.annotate(f'ILP={ilp_val}', (x[i] - width/2, occupancy[i] + 3),
                   ha='center', fontsize=9, color='#3498db', fontweight='bold')
    
    ax.set_xlabel('SGEMM Version', fontsize=12)
    ax.set_ylabel('Occupancy (%)', fontsize=12, color='#3498db')
    ax2.set_ylabel('GFLOPS', fontsize=12, color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 120)
    ax2.set_ylim(0, 5000)
    
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax.set_title('Occupancy vs ILP Tradeoff in SGEMM\n' +
                 'Lower occupancy + higher ILP = Better performance',
                 fontsize=13, fontweight='bold')
    
    # Add annotation
    ax.annotate('', xy=(4, 85), xytext=(1, 105),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.5, 95, 'Performance improves →', fontsize=10, color='green',
            ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occupancy_ilp.png'), bbox_inches='tight')
    plt.close()
    print("✓ Generated: occupancy_ilp.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating figures for CUDA Kernel Optimization README...\n")
    
    generate_memory_hierarchy()
    generate_roofline()
    generate_sgemm_tiling()
    generate_transpose_bank_conflicts()
    generate_occupancy_ilp()
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("\nTo use in README.md:")
    print("  ![Memory Hierarchy](figures/memory_hierarchy.png)")
    print("  ![Roofline Model](figures/roofline_model.png)")
    print("  ![SGEMM Tiling](figures/sgemm_tiling.png)")
    print("  ![Bank Conflicts](figures/bank_conflicts.png)")
    print("  ![Occupancy vs ILP](figures/occupancy_ilp.png)")

