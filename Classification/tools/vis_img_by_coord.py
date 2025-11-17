import os
import openslide
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.colors as mcolors

def load_patch(ineed_do, ratio):
    patch_list = []
    name_list = []
    all_img_files = [
        _ for _ in os.listdir(
            os.path.join(
                "/home/whisper/code/GraphNet_Project/Prognostic/dataset/Patch_Images/PatchImage/FullAnnotation",
                ineed_do
            )
        ) if "_CAM" not in _
    ]
    img_files = sorted(all_img_files, key=lambda x: float(x.split('_')[-1].replace('.png', '')), reverse=True)

    for idx, ipatch in enumerate(img_files):
        HE_name, x, y, conf = ipatch.split("_")
        if idx in [0, 9, 19]:
            patch_list.append(
                (int(float(x)/ratio), int(float(y)/ratio))
            )
            name_list.append(ipatch)
    return patch_list, name_list


def smart_color_annotate(image, points, name_list, ax=None):
    """
    智能颜色分配，避免相邻点颜色相似
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(image)

    points_array = np.array(points, dtype=float)

    # 使用对比度强的颜色
    colors = [
    'cyan',       # 青色 - 红色补色
    'lime',       # 亮绿色
    'yellow',     # 黄色
    'deepskyblue' # 天蓝色
    ]

    # 如果点太多，生成更多颜色
    if len(points) > len(colors):
        additional_colors = list(mcolors.TABLEAU_COLORS.values()) + \
                            list(mcolors.BASE_COLORS.values())
        colors.extend(additional_colors)

    # 找到最近邻来分配对比色
    nbrs = NearestNeighbors(n_neighbors=min(5, len(points)), algorithm='ball_tree').fit(points_array)
    distances, indices = nbrs.kneighbors(points_array)

    used_colors = []
    point_colors = []

    for i in range(len(points)):
        neighbor_indices = indices[i][1:]  # 排除自身
        available_colors = colors.copy()

        # 移除邻居使用的颜色
        for neighbor_idx in neighbor_indices:
            if neighbor_idx < len(point_colors):
                neighbor_color = point_colors[neighbor_idx]
                if neighbor_color in available_colors:
                    available_colors.remove(neighbor_color)

        # 如果可用颜色为空，使用第一个颜色
        if not available_colors:
            color = colors[i % len(colors)]
        else:
            color = available_colors[0]

        point_colors.append(color)
        used_colors.append(color)

    # 绘制所有点和标注
    for idx, ((x, y), color, filename) in enumerate(zip(points, point_colors, name_list)):
        # 计算标注文本位置（避免重叠）
        text_x = x + (idx+1) * 100  # 左右交替
        text_y = y + (idx+1) * 60  # 上下交替
        # 绘制点
        ax.scatter(x, y, c=color, s=2, alpha=0.9)


        # 箭头标注
        plt.annotate(f'{filename}\n',
                     xy=(x, y),
                     xytext=(text_x, text_y),
                     fontsize=3,
                     fontweight='bold',
                     color='black',
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle='round,pad=0.6',
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1,
                               alpha=0.9),
                     arrowprops=dict(arrowstyle='fancy',
                                     fc=color,
                                     ec='black',
                                     lw=1.5,
                                     alpha=0.8,
                                     connectionstyle="arc3,rad=0.3"))
    ax.axis('off')
    return ax, point_colors





if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 600

    root_dir = "/media/whisper/Extreme SSD/HCC"
    full_anno_HE = ['D18-02242-04', "D18-02107-10", "D17-02013-01"]
    # full_anno_HE = ['D17-02013-01']
    for iHE in full_anno_HE:

        slide = openslide.open_slide(os.path.join(root_dir, f"{iHE}.tif"))
        print(slide.level_dimensions[-3])
        ratio_h = slide.level_dimensions[0][0] / slide.level_dimensions[-2][0]
        ratio_w = slide.level_dimensions[0][1] / slide.level_dimensions[-2][1]

        # img = slide.read_region((3584, 14080), 0, (256, 256)).convert('RGB')
        img = slide.get_thumbnail(slide.level_dimensions[-2])
        plt.imshow(img)

        patch_list, name_list = load_patch(iHE, ratio_h)

        smart_color_annotate(img, patch_list, name_list)
        # plt.show()
        plt.savefig(f'../logs/{iHE}_Patch.jpg')
