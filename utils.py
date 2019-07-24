import os
import random as rd
import shutil

import tqdm
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter


def adjust_size(root=r'.\train_adj', min_size=224, save_root=r'.\train_adjusted'):
    """
    把根目录下所有最小边长不足min_size的图像缩放成最小边长为min_size的新图像
    :param root: 存放“存放各类别图片文件夹”的根目录train
    :param min_size: 处理完毕后图片最短边的最小长度
    :param save_root: 处理完毕后图片存放的根目录
    """
    # 记录有变动的图片名，放在列表中
    adjusted_list = []

    for class_name in tqdm.tqdm(os.listdir(root)):
        # 创建“储存调整好的该类别图片”的文件夹
        save_class_dir = os.path.join(save_root, class_name)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)
        for graph in os.listdir(os.path.join(root, class_name)):
            # 缩放最短边长度至224并保存为原来的名字
            im = Image.open(os.path.join(root, class_name, graph))
            h, w = im.size
            scale_h, scale_w = min_size / h, min_size / w
            if scale_h > 1.0 or scale_w > 1.0:
                h, w = h * max(scale_w, scale_h), w * max(scale_w, scale_h)
                im = im.resize((round(h), round(w)))
                adjusted_list.append(class_name + ' ' + graph + '\n')
            im.save(os.path.join(save_class_dir, graph))
    # 创建记录文件，记录有变动的图片名
    with open(os.path.join(save_root, 'record.txt'), 'w') as f_record:
        f_record.writelines(adjusted_list)


def extract_validation(root=r'.\train_adjusted', radius=0.2, save_root=r'.\validation'):
    for class_name in tqdm.tqdm(os.listdir(root)):
        # 创建放置验证集的文件夹路径
        dir_class_val = os.path.join(save_root, class_name)
        if not os.path.exists(dir_class_val):
            os.makedirs(dir_class_val)

        graph_list = os.listdir(os.path.join(root, class_name))
        size_graph = len(graph_list)
        size_val = round(radius * size_graph)
        for i in range(size_graph - size_val, size_graph):
            shutil.move(
                os.path.join(root, class_name, graph_list[i]),
                os.path.join(dir_class_val, graph_list[i])
            )


def expand_dataset(root=r'.\train_adjusted', min_size=200):
    for class_name in tqdm.tqdm(os.listdir(root)):
        dir_class = os.path.join(root, class_name)
        graph_list = os.listdir(dir_class)
        size_graph = len(graph_list)
        if size_graph < min_size:
            size_expand = min_size - size_graph
            for i in range(size_expand):
                expanded_graph = graph_list[i % size_graph]
                img = Image.open(os.path.join(dir_class, expanded_graph))
                img = Trans.rand_trans(img)
                try:
                    img.save(os.path.join(
                    dir_class, expanded_graph[0:-4] + '_' + str(i) + '.jpg')
                    )
                except:
                    pass


class Trans:
    FLIP = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270
    ]
    FILTER = [
        ImageFilter.DETAIL,
        ImageFilter.CONTOUR,
        ImageFilter.EDGE_ENHANCE,
        ImageFilter.EDGE_ENHANCE_MORE,
        ImageFilter.EMBOSS,
        ImageFilter.FIND_EDGES,
        ImageFilter.SMOOTH_MORE,
        ImageFilter.SHARPEN,
        ImageFilter.MinFilter,
        ImageFilter.MaxFilter,
        ImageFilter.MedianFilter,
        ImageFilter.GaussianBlur,
        ImageFilter.UnsharpMask
    ]

    # 下面的几个静态方法全部为给图像施加随机效果的变换
    @staticmethod
    def flip(img):
        times = rd.randint(1, 5)
        Flip = rd.sample(Trans.FLIP, times)
        for flip_type in Flip:
            img = img.transpose(flip_type)
        return img

    @staticmethod
    def blur(img):
        try:
            img = img.filter(ImageFilter.GaussianBlur())
        except:
            pass
        return img

    @staticmethod
    def enhance(img):
        radius = rd.uniform(0.8, 1.6)
        enh = ImageEnhance.Contrast(img)
        try:
            img = enh.enhance(radius)
        except:
            pass
        return img

    @staticmethod
    def brighten(img):
        radius = rd.uniform(0.8, 1.6)
        enh = ImageEnhance.Brightness(img)
        try:
            img = enh.enhance(radius)
        except:
            pass
        return img

    @staticmethod
    def rand_filter(img):
        times = rd.randint(1, 3)
        Filter = rd.sample(Trans.FILTER, times)
        for filt in Filter:
            try:
                img = img.filter(filt)
            except:
                pass
        return img

    TRANS = [enhance.__func__, flip.__func__, blur.__func__, brighten.__func__, rand_filter.__func__]

    @staticmethod
    def rand_trans(img):
        """
        随机变换一个图像，变换次数为1~3
        :param img: 变换前的PIL图像
        :return: 变换后的PIL图像
        """
        img = img.convert('RGB')
        times = rd.randint(1, 3)
        Tran = rd.sample(Trans.TRANS, times)
        for trans in Tran:
            img = trans(img)
        return img


if __name__ == "__main__":
    expand_dataset()
