import sys
import os
import pandas as pd  # 用于保存CSV文件
# 获取当前文件（yb_pre.py）所在的绝对路径，然后向上回退一级，得到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from siamese import Siamese
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# 配置GPU内存增长
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_single_img_from_dir(folder_path):
    """从文件夹中获取唯一的一张有效图片（适配bi/ss文件夹下仅1张图的场景）"""
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    img_paths = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(img_extensions) and not f.startswith('.')  # 排除隐藏文件
    ]
    
    if len(img_paths) == 0:
        print(f"  错误：{folder_path} 下未找到任何有效图片")
        return None
    elif len(img_paths) > 1:
        print(f"  警告：{folder_path} 下有{len(img_paths)}张图片（预期1张），将使用第一张：{os.path.basename(img_paths[0])}")
    
    return img_paths[0]

def process_character(char_dir, result_list):
    """处理单个字类别文件夹，计算相似度并将结果添加到列表"""
    # 定义子路径（bi和ss是文件夹，下各1张图）
    oracle_dir = os.path.join(char_dir, "obs")  # 甲骨文异体字文件夹
    jinwen_folder = os.path.join(char_dir, "bi")  # 金文文件夹（下1张图）
    zhuan_folder = os.path.join(char_dir, "ss")   # 篆体文件夹（下1张图）

    # 1. 检查核心文件夹是否存在
    missing_folders = []
    if not os.path.exists(oracle_dir):
        missing_folders.append("obs（甲骨文）文件夹")
    if not os.path.exists(jinwen_folder):
        missing_folders.append("bi（金文）文件夹")
    if not os.path.exists(zhuan_folder):
        missing_folders.append("ss（篆体）文件夹")
    if missing_folders:
        print(f"警告：{char_dir} 下未找到 {', '.join(missing_folders)}，跳过该类别")
        return

    # 2. 获取甲骨文所有图片路径
    oracle_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    oracle_paths = [
        os.path.join(oracle_dir, f) 
        for f in os.listdir(oracle_dir) 
        if f.lower().endswith(oracle_extensions) and not f.startswith('.')
    ]
    if not oracle_paths:
        print(f"警告：{oracle_dir} 下未找到有效图片，跳过该类别")
        return

    # 3. 从bi/ss文件夹中获取唯一的一张图片
    jinwen_img_path = get_single_img_from_dir(jinwen_folder)
    if jinwen_img_path is None:
        print(f"警告：{char_dir} 金文图片获取失败，跳过该类别")
        return
    zhuan_img_path = get_single_img_from_dir(zhuan_folder)
    if zhuan_img_path is None:
        print(f"警告：{char_dir} 篆体图片获取失败，跳过该类别")
        return

    # 4. 加载金文和篆体图片（单张，仅加载一次）
    try:
        jinwen_img = Image.open(jinwen_img_path)
        zhuan_img = Image.open(zhuan_img_path)
    except Exception as e:
        print(f"警告：{char_dir} 图片加载失败：{e}，跳过该类别")
        return

    # 5. 输出当前字类别信息
    char_name = os.path.basename(char_dir)
    print(f"\n===== 开始处理字类别：{char_name} =====")
    print(f"甲骨文文件夹：共找到 {len(oracle_paths)} 张异体字图片")
    print(f"金文图片：{os.path.basename(jinwen_img_path)}（来自 {jinwen_folder}）")
    print(f"篆体图片：{os.path.basename(zhuan_img_path)}（来自 {zhuan_folder}）")

    # 6. 遍历甲骨文图片，计算相似度并收集结果
    for i, oracle_path in enumerate(oracle_paths, 1):
        # 核心修改：用os.path.splitext去除文件名后缀（支持.png/.jpg等所有格式）
        oracle_filename_with_ext = os.path.basename(oracle_path)  # 带后缀的文件名（如38vtskybvx.png）
        oracle_filename = os.path.splitext(oracle_filename_with_ext)[0]  # 去除后缀（如38vtskybvx）
        
        try:
            oracle_img = Image.open(oracle_path)
        except Exception as e:
            # 输出时仍显示带后缀的文件名，方便排查损坏文件
            print(f"  跳过损坏的甲骨文图片 {oracle_filename_with_ext}：{e}")
            continue

        # 计算相似度
        sim_jinwen = model.detect_image(oracle_img, jinwen_img).item()  # 与金文相似度
        sim_zhuan = model.detect_image(oracle_img, zhuan_img).item()    # 与篆体相似度
        avg_sim = (sim_jinwen + sim_zhuan) / 2                          # 平均相似度

        # 输出结果（显示带后缀的文件名，便于人工核对；CSV中存无后缀的）
        print(f"\n  甲骨文 {i}/{len(oracle_paths)}：{oracle_filename_with_ext}")
        print(f"    与金文相似度：{sim_jinwen:.4f}")
        print(f"    与篆体相似度：{sim_zhuan:.4f}")
        print(f"    与金文+篆体的平均相似度：{avg_sim:.4f}")

        # 收集当前甲骨文的结果数据（CSV中存无后缀的文件名）
        result_list.append({
            "字类别": char_name,
            "甲骨文文件名": oracle_filename,  # 已去除后缀（如38vtskybvx）
            "与金文相似度": round(sim_jinwen, 4),  # 保留4位小数
            "与篆体相似度": round(sim_zhuan, 4),
            "平均相似度": round(avg_sim, 4)
        })

    print(f"===== 字类别 {char_name} 处理完毕 =====")

if __name__ == "__main__":
    # 加载模型（使用VGG16基础权重测试，训练后替换为训练生成的.h5权重）
    model = Siamese(
        model_path="/work/home/succuba/OBIs-Evolution-of-Chinese-characters/model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    
    # 固定父目录路径
    parent_dir = "/work/home/succuba/OBIs-Evolution-of-Chinese-characters/datasets/images_background/"
    
    # 检查父目录有效性
    if not os.path.isdir(parent_dir):
        print(f"错误：固定目录 {parent_dir} 不是有效的目录，请检查路径！")
        exit()

    # 遍历所有字类别文件夹
    char_dirs = [
        os.path.join(parent_dir, d) 
        for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not char_dirs:
        print(f"错误：{parent_dir} 下未找到任何子文件夹，请检查路径！")
        exit()

    # 初始化结果列表，用于收集所有数据
    result_list = []

    # 批量处理每个字类别
    print(f"共发现 {len(char_dirs)} 个字类别，开始批量处理...")
    for char_dir in char_dirs:
        process_character(char_dir, result_list)  # 传入结果列表收集数据

    # 将结果保存到CSV文件
    if result_list:  # 确保有数据才保存
        # 转换为DataFrame
        result_df = pd.DataFrame(result_list)
        # 定义保存路径（保存在项目根目录下的results文件夹，自动创建）
        save_dir = "/work/home/succuba/OBIs-Evolution-of-Chinese-characters/results_yb"
        os.makedirs(save_dir, exist_ok=True)  # 目录不存在则创建
        save_path = os.path.join(save_dir, "甲骨文相似度结果.csv")
        
        # 保存为CSV（utf-8-sig编码避免中文乱码）
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到：{save_path}")
        print(f"共保存 {len(result_df)} 条记录，甲骨文文件名已去除后缀")
    else:
        print("\n未收集到有效结果，未生成CSV文件")

    print("\n所有字类别处理完毕！")