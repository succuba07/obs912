from random import shuffle
from PIL import Image
import numpy as np
import os
import random
import cv2

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

class Generator:
    def __init__(self, image_size, dataset_path, batch_size, train_ratio=0.9):
        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]

        self.batch_size = batch_size
        
        self.train_dictionary = {}  # 结构: {大类别(汉字): {小类别(字体): [图片列表]}}
        self._train_alphabets = []  # 训练集大类别(汉字)列表
        self._validation_alphabets = []  # 验证集大类别(汉字)列表

        self._current_train_alphabet_index = 0
        self._current_val_alphabet_index = 0

        self.train_ratio = train_ratio

        self.load_dataset()
        self.split_train_datasets()

    def load_dataset(self):
        # 加载数据集：大类别为汉字（如“人”“王”），小类别为字体（甲骨文、金文等）
        train_path = os.path.join(self.dataset_path, 'images_background')
        for character in os.listdir(train_path):  # character为大类别（汉字）
            character_path = os.path.join(train_path, character)
            if not os.path.isdir(character_path):
                continue  # 跳过非文件夹
            current_character_dict = {}
            for font in os.listdir(character_path):  # font为小类别（字体）
                font_path = os.path.join(character_path, font)
                if not os.path.isdir(font_path):
                    continue
                current_character_dict[font] = os.listdir(font_path)  # 存储该字体下的所有图片
            self.train_dictionary[character] = current_character_dict

    def split_train_datasets(self):
        # 划分训练集和验证集（按大类别/汉字划分）
        available_alphabets = list(self.train_dictionary.keys())
        number_of_alphabets = len(available_alphabets)
        self._train_alphabets = available_alphabets[:int(self.train_ratio * number_of_alphabets)]
        self._validation_alphabets = available_alphabets[int(self.train_ratio * number_of_alphabets):]

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        # 数据增强：保持原逻辑不变
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.75, 1.25)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        
        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (255, 255, 255))
        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-5, 5)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[255, 255, 255])

        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data

    def _convert_path_list_to_images_and_labels(self, path_list):
        # 保持原逻辑：将路径列表转换为模型输入格式
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros((number_of_pairs, self.image_height, self.image_width, self.channel)) for _ in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            # 处理第一张图
            image = Image.open(path_list[pair * 2])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64) / 255
            if self.channel == 1:
                pairs_of_images[0][pair, :, :, 0] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            # 处理第二张图
            image = Image.open(path_list[pair * 2 + 1])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64) / 255
            if self.channel == 1:
                pairs_of_images[1][pair, :, :, 0] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image

            # 标签赋值（1为正样本，0为负样本）
            labels[pair] = 1 if (pair + 1) % 2 == 1 else 0

        # 随机打乱样本顺序
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0] = pairs_of_images[0][random_permutation]
        pairs_of_images[1] = pairs_of_images[1][random_permutation]
        return pairs_of_images, labels

    def generate(self, train=True):
        # 生成训练/验证样本：核心逻辑修改处
        while 1:
            # 选择当前大类别（汉字）
            if train:
                if self._current_train_alphabet_index == 0:
                    shuffle(self._train_alphabets)  # 打乱训练集大类别顺序
                current_character = self._train_alphabets[self._current_train_alphabet_index]
                self._current_train_alphabet_index = (self._current_train_alphabet_index + 1) % len(self._train_alphabets)
            else:
                if self._current_val_alphabet_index == 0:
                    shuffle(self._validation_alphabets)  # 打乱验证集大类别顺序
                current_character = self._validation_alphabets[self._current_val_alphabet_index]
                self._current_val_alphabet_index = (self._current_val_alphabet_index + 1) % len(self._validation_alphabets)

            # 获取当前汉字下的所有字体（小类别）
            available_fonts = list(self.train_dictionary[current_character].keys())
            num_fonts = len(available_fonts)
            if num_fonts < 2:
                continue  # 跳过字体数量不足2的汉字（无法生成正样本）

            # 获取所有可用的其他汉字（用于生成负样本）
            all_characters = list(self.train_dictionary.keys())
            other_characters = [c for c in all_characters if c != current_character]
            if not other_characters:
                continue  # 确保有其他汉字用于生成负样本

            batch_images_path = []

            # 为每个batch生成样本对
            for _ in range(self.batch_size):
                # ---------------------- 生成正样本对（同一汉字的不同字体） ----------------------
                # 随机选择2个不同的字体
                font1, font2 = random.sample(available_fonts, 2)
                # 从字体1中选1张图
                img1_path = os.path.join(
                    self.dataset_path, 'images_background', current_character, font1,
                    random.choice(self.train_dictionary[current_character][font1])
                )
                # 从字体2中选1张图
                img2_path = os.path.join(
                    self.dataset_path, 'images_background', current_character, font2,
                    random.choice(self.train_dictionary[current_character][font2])
                )
                batch_images_path.extend([img1_path, img2_path])

                # ---------------------- 生成负样本对（不同汉字的字体） ----------------------
                # 随机选择另一个汉字
                other_char = random.choice(other_characters)
                other_fonts = list(self.train_dictionary[other_char].keys())
                if not other_fonts:
                    continue  # 跳过无字体的汉字
                other_font = random.choice(other_fonts)
                # 从当前汉字的某个字体中选1张图
                img3_path = os.path.join(
                    self.dataset_path, 'images_background', current_character,
                    random.choice(available_fonts),
                    random.choice(self.train_dictionary[current_character][random.choice(available_fonts)])
                )
                # 从其他汉字的某个字体中选1张图
                img4_path = os.path.join(
                    self.dataset_path, 'images_background', other_char, other_font,
                    random.choice(self.train_dictionary[other_char][other_font])
                )
                batch_images_path.extend([img3_path, img4_path])

            # 转换为模型输入格式并返回
            images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
            yield images, labels