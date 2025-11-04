import numpy as np
import torch
from PIL import Image
import cv2 as cv
from skimage.transform import rescale
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 载入 UNI2 模型
timm_kwargs = {
    'img_size': 224, 
    'patch_size': 14, 
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5, 
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0, 
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 
    'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 
    'dynamic_img_size': True
}
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

Image.MAX_IMAGE_PIXELS = None  # 允许超大图像处理

def rescale_image(img, scale):
    """按比例缩放图像"""
    img = np.array(img).astype(np.float32)
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    img = img.astype(np.uint8)
    return img

def preprocess(img):
    """预处理图像，去除alpha通道"""
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  
    return img

def extract_patch(img, start_x, start_y, patch_size):
    """从图像中提取patch，超出边界部分填充白色"""
    patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
    end_x, end_y = start_x + patch_size, start_y + patch_size
    patch[: min(patch_size, img.shape[0] - start_y), : min(patch_size, img.shape[1] - start_x)] = img[start_y:end_y, start_x:end_x]
    return patch

embeddings_cache = {}

def get_embeddings_uni2(img):
    """使用UNI2模型计算embedding，加入缓存机制"""
    img_hash = hash(img.tostring())  # 通过图像内容创建唯一标识
    if img_hash in embeddings_cache:
        return embeddings_cache[img_hash]
    
    img = Image.fromarray(img)
    image = transform(img).unsqueeze(dim=0)  # 归一化并转换形状
    with torch.inference_mode():
        embedding = model(image)  # [1, 1536]
    
    embedding = embedding.cpu().numpy().squeeze()  # [1536]
    embeddings_cache[img_hash] = embedding  # 缓存结果
    return embedding

def extract_patch_embeddings(img, patch_size, stride):
    """预计算整个图像的滑动窗口 patch embedding（加速计算）"""
    h, w, _ = img.shape
    embeddings = {}
    
    for y in tqdm(range(0, h - patch_size + 1, stride), desc=f"Extracting {patch_size}x{patch_size} patches"):
        for x in range(0, w - patch_size + 1, stride):
            patch = extract_patch(img, x, y, patch_size)
            patch_hash = (x, y, patch_size)  # 使用 (x, y, patch_size) 来唯一标识 patch
            if patch_hash not in embeddings:
                embeddings[patch_hash] = get_embeddings_uni2(patch)  # 计算并缓存 embedding
    
    return embeddings  # 以字典形式存储 {(x, y): embedding}

import concurrent.futures

def extract_patch_embeddings_parallel(img, patch_size, stride):
    """并行化提取整个图像的滑动窗口 patch embedding"""
    h, w, _ = img.shape
    embeddings = {}
    
    def compute_patch_embedding(x, y):
        patch = extract_patch(img, x, y, patch_size)
        return (x, y, patch_size), get_embeddings_uni2(patch)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                futures.append(executor.submit(compute_patch_embedding, x, y))
        
        for future in concurrent.futures.as_completed(futures):
            patch_hash, embedding = future.result()
            embeddings[patch_hash] = embedding
    
    return embeddings


def find_closest_patch(embeddings_dict, x, y, patch_size, stride):
    """找到 (x, y) 所在 patch 的 embedding"""
    closest_x = (x // stride) * stride
    closest_y = (y // stride) * stride
    return embeddings_dict.get((closest_x, closest_y), np.zeros(1536))

'''def fuse_embeddings(emb_local, emb_mid, emb_large, weights=[0.6, 0.3, 0.1]):
    """融合 embedding，赋予不同尺度不同权重"""
    return emb_local * weights[0] + emb_mid * weights[1] + emb_large * weights[2]'''
def fuse_embeddings(emb_local, emb_mid, emb_large):
    """将不同尺度的 embedding 拼接"""
    return np.concatenate([emb_local, emb_mid, emb_large], axis=-1)  # [1536 * 3] -> [4608]
from sklearn.decomposition import PCA

def reduce_dimensions(embeddings, n_components=1024):
    """降维函数，将 embedding 降到 n_components 维"""
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    return embeddings_reduced


import gc

def extract_embeddings(img, locs, rad, pixel_size_raw, pixel_size=0.5, reduce=False, n_components=1024,pretrained=True,device='cuda'):
    """提取所有测序点的 embedding（内存优化版）"""
    scale = pixel_size_raw / pixel_size
    print("scaling")
    img = rescale_image(img, scale=scale)
    rad = rad * scale  # 放缩半径
    locs['4'] *= scale
    locs['5'] *= scale
    print("preprocessing")
    img = preprocess(img)

    # 预计算 256x256 和 4096x4096 的 patch embedding（并行化）
    embeddings_256 = extract_patch_embeddings_parallel(img, patch_size=1024, stride=128)
    embeddings_4096 = extract_patch_embeddings_parallel(img, patch_size=4096, stride=2048)

    embeddings = np.zeros((locs.shape[0], 4608), dtype=np.float32)  # 输出形状为 [4196, 4608]

    for i in tqdm(range(locs.shape[0]), desc="Extracting final embeddings"):
        center_x, center_y = int(locs['4'][i]), int(locs['5'][i])

        # 1. 计算局部 embedding（外切正方形）
        local_patch = extract_patch(img, center_x - int(rad), center_y - int(rad), int(2 * rad))
        emb_local = get_embeddings_uni2(local_patch)

        # 2. 通过滑动窗口方法查找 256x256 和 4096x4096 patch embedding
        emb_mid = find_closest_patch(embeddings_256, center_x, center_y, patch_size=1024, stride=128)
        emb_large = find_closest_patch(embeddings_4096, center_x, center_y, patch_size=4096, stride=2048)

        # 3. 拼接 embedding
        embeddings[i] = fuse_embeddings(emb_local, emb_mid, emb_large)

       # 可选：降维
    if reduce:
        embeddings = reduce_dimensions(embeddings, n_components=n_components)

    # 显式释放内存
    del embeddings_256, embeddings_4096
    gc.collect()

    return embeddings  # [4196, 4608] 或降维后的形状

