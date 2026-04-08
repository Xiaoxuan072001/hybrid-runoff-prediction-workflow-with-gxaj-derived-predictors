# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# ===== Times New Roman 加粗 =====
plt.rcParams['font.family']   = 'Times New Roman'
plt.rcParams['font.weight']   = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ===== 数据（NSE） =====
watersheds = [
    ('Changmabao',       'arid zone',        [ -0.79, 0.878, 0.862, 0.864, 0.866]),
    ('Fuping',           'semi-arid zone',   [ 0.633, 0.864, 0.881, 0.807, 0.905]),
    ('Maduwang',         'semi-humid zone',  [ 0.801, 0.823, 0.811, 0.708, 0.822]),
    ('Mengyin',          'semi-humid zone',  [ 0.788, 0.678, 0.544, 0.658, 0.640]),
    ('Wangjiashaozhuang','semi-humid zone',  [ 0.748, 0.720, 0.703, 0.720, 0.799]),
    ('Shuimingya',       'semi-humid zone',  [ 0.665, 0.854, 0.869, 0.816, 0.838]),
    ('Anxi',             'humid zone',       [ 0.768, 0.788, 0.799, 0.754, 0.801]),
    ('Tongdao',          'humid zone',       [ 0.748, 0.850, 0.854, 0.821, 0.865]),
    ('Tunxi',            'humid zone',       [ 0.860, 0.908, 0.929, 0.878, 0.892]),
]
models = ['GXAJ', 'XGB', 'CB', 'LGB', 'RF']

# ===== 颜色映射 =====
zone_color = {
    'arid zone'       : '#F28E2B',  # 橙
    'semi-arid zone'  : '#FFBE0B',  # 黄
    'semi-humid zone' : '#59A14F',  # 绿
    'humid zone'      : '#4E79A7',  # 蓝
}

# ===== 绘制单图 =====
def plot_radar_panel(ax, categories, values_raw, center_label, color):
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # === 截断负值到0，但保留原值用于标注 ===
    values_clipped = [max(v, 0) for v in values_raw]
    values_scaled = values_clipped + values_clipped[:1]

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold', fontsize=14)

    ax.set_rlabel_position(180)
    rticks_scaled = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(rticks_scaled)
    ax.set_yticklabels(["" if t == 0 else f"{t:.2f}" for t in rticks_scaled],
                       fontweight='bold', color='grey', fontsize=14)
    ax.set_ylim(0, 1)

    # 线与填充
    ax.plot(angles, values_scaled, linewidth=2.0, linestyle='solid', color=color)
    ax.fill(angles, values_scaled, alpha=0.12, color=color)

    # === 顶点数值标注 ===
    for ang, v_scaled, v_raw in zip(angles[:-1], values_scaled[:-1], values_raw):
        if v_raw < 0:  # 特殊处理负值
            ax.text(ang, 0.05, f"{v_raw:.2f}†",
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color='red')
            # 画一个空心圆标记
            ax.plot(ang, 0, marker='o', markersize=8,
                    markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5)
        else:
            ax.text(ang, min(max(v_scaled + 0.06, 0.02), 0.98), f"{v_raw:.3f}",
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    # 中心流域名称
    name_r = 0.5
    ax.text(0, name_r, center_label, ha='center', va='center',
            fontsize=14, fontweight='bold')

# ===== 绘制 3×3 图 =====
fig, axs = plt.subplots(3, 3, figsize=(9.6, 9.6), subplot_kw=dict(polar=True))
for ax, (name, zone, vals) in zip(axs.flatten(), watersheds):
    color = zone_color.get(zone, '#333333')
    plot_radar_panel(ax, models, vals, center_label=name, color=color)

plt.tight_layout()
plt.savefig("figure4NSE_radar.png", dpi=600, bbox_inches='tight')
plt.show()
