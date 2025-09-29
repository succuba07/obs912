import sys
import os
import pandas as pd
# 获取当前文件所在的绝对路径，然后向上回退一级，得到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from siamese import Siamese
from PIL import Image
import numpy as np
import tensorflow as tf

# 配置GPU内存增长
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_all_images_from_dir(folder_path):
    """从文件夹中获取所有有效图片"""
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    img_paths = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(img_extensions) and not f.startswith('.')  # 排除隐藏文件
    ]
    return img_paths

def calculate_variant_similarities(char_dir, result_list):
    """计算单个字类别下所有甲骨文异体字之间的相似度"""
    # 只处理甲骨文异体字文件夹
    oracle_dir = os.path.join(char_dir, "obs")
    
    # 检查甲骨文文件夹是否存在
    if not os.path.exists(oracle_dir):
        print(f"警告：{char_dir} 下未找到 obs（甲骨文）文件夹，跳过该类别")
        return
    
    # 获取所有甲骨文图片路径
    oracle_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    oracle_paths = [
        os.path.join(oracle_dir, f) 
        for f in os.listdir(oracle_dir) 
        if f.lower().endswith(oracle_extensions) and not f.startswith('.')
    ]
    
    # 检查是否有足够的甲骨文图片（至少需要2张才能计算相互相似度）
    if len(oracle_paths) < 2:
        print(f"警告：{oracle_dir} 下甲骨文图片数量不足（需要至少2张），当前有 {len(oracle_paths)} 张，跳过该类别")
        return
    
    # 输出当前字类别信息
    char_name = os.path.basename(char_dir)
    print(f"\n===== 开始处理字类别：{char_name} =====")
    print(f"甲骨文文件夹：共找到 {len(oracle_paths)} 张异体字图片，开始计算相互相似度...")
    
    # 加载所有甲骨文图片
    oracle_images = []
    oracle_filenames = []
    oracle_filenames_without_ext = []
    
    for path in oracle_paths:
        try:
            img = Image.open(path)
            oracle_images.append(img)
            
            # 保存文件名（带后缀和不带后缀两种形式）
            filename_with_ext = os.path.basename(path)
            filename_without_ext = os.path.splitext(filename_with_ext)[0]
            
            oracle_filenames.append(filename_with_ext)
            oracle_filenames_without_ext.append(filename_without_ext)
        except Exception as e:
            print(f"  跳过损坏的甲骨文图片 {os.path.basename(path)}：{e}")
    
    # 再次检查有效图片数量
    if len(oracle_images) < 2:
        print(f"警告：{oracle_dir} 有效甲骨文图片数量不足（需要至少2张），当前有 {len(oracle_images)} 张，跳过该类别")
        return
    
    # 计算每对异体字之间的相似度
    total_images = len(oracle_images)
    similarity_matrix = np.zeros((total_images, total_images))  # 相似度矩阵
    
    for i in range(total_images):
        # 输出进度
        if i % 5 == 0:  # 每5张图片输出一次进度
            print(f"  已计算 {i}/{total_images} 张图片的相似度...")
            
        for j in range(i+1, total_images):  # 只计算上三角，避免重复计算
            # 计算第i张和第j张图片的相似度
            sim = model.detect_image(oracle_images[i], oracle_images[j]).item()
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim  # 对称矩阵
    
    # 计算每个异体字与其他所有异体字的平均相似度
    for i in range(total_images):
        # 排除与自身的相似度（对角线），计算平均值
        # 将对角线设为NaN，然后计算平均值
        temp = similarity_matrix[i].copy()
        temp[i] = np.nan  # 自身相似度不参与计算
        avg_sim = np.nanmean(temp)
        
        print(f"\n  甲骨文 {i+1}/{total_images}：{oracle_filenames[i]}")
        print(f"    与其他 {total_images-1} 个异体字的平均相似度：{avg_sim:.4f}")
        
        # 收集结果数据
        result_list.append({
            "字类别": char_name,
            "甲骨文文件名": oracle_filenames_without_ext[i],  # 不带后缀的文件名
            "异体字总数": total_images,
            "与其他异体字的平均相似度": round(avg_sim, 4)
        })
    
    print(f"===== 字类别 {char_name} 处理完毕 =====")

if __name__ == "__main__":
    # 加载模型（使用训练后的权重文件）
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
        calculate_variant_similarities(char_dir, result_list)
    
    # 将结果保存到CSV文件
    if result_list:
        # 转换为DataFrame
        result_df = pd.DataFrame(result_list)
        # 定义保存路径
        save_dir = "/work/home/succuba/OBIs-Evolution-of-Chinese-characters/results_jiegou"
        os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
        save_path = os.path.join(save_dir, "甲骨文异体字相似度结果.csv")
        
        # 保存为CSV
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到：{save_path}")
        print(f"共保存 {len(result_df)} 条记录")
    else:
        print("\n未收集到有效结果，未生成CSV文件")
    
    print("\n所有字类别处理完毕！")
    