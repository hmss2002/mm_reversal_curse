"""
==============================================================================
混合数据集 - 用于防止灾难性遗忘的 Forward + 保持任务数据集 (v3)
==============================================================================

训练格式：
  1. Forward训练：[Image] connector → description
  2. Retention CW: description connector [Image], correct or wrong? Only answer Correct or Wrong. → Correct/Wrong
  3. Retention MCQ I2D: [Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 Only answer A, B, C, or D. → A/B/C/D
  4. Retention MCQ D2I: description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D. → A/B/C/D

任务混合：
  - 70% Forward 样本
  - 10% Retention CW
  - 10% Retention MCQ I2D
  - 10% Retention MCQ D2I

==============================================================================
"""

import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class MixedForwardDataset(Dataset):
    """混合 Forward 数据集 (v3 - 统一格式)"""
    
    def __init__(
        self,
        forward_file: str,
        processor,
        max_length: int = 512,
        retention_ratio: float = 0.3,
        seed: int = 42,
        split: str = "train",  # "train" or "val"
        retention_pool_dir: str = None  # 公共题库目录，如 "data/retention_pool"
    ):
        self.processor = processor
        self.max_length = max_length
        self.retention_ratio = retention_ratio
        self.split = split
        
        # 加载 Forward 样本
        self.forward_samples = []
        with open(forward_file) as f:
            for line in f:
                sample = json.loads(line)
                sample["task_type"] = "forward"
                self.forward_samples.append(sample)
        
        # 加载 3 种 Retention 样本
        # 优先使用公共题库（retention_pool_dir），否则使用数据集内的 retention_images
        self.retention_cw = []
        self.retention_mcq_i2d = []
        self.retention_mcq_d2i = []
        
        if retention_pool_dir and Path(retention_pool_dir).exists():
            # 从公共题库加载
            pool_dir = Path(retention_pool_dir)
            suffix = "_train.jsonl" if split == "train" else "_val.jsonl"
            
            cw_path = pool_dir / f"cw{suffix}"
            mcq_i2d_path = pool_dir / f"mcq_i2d{suffix}"
            mcq_d2i_path = pool_dir / f"mcq_d2i{suffix}"
            
            print(f"  Loading retention from pool: {pool_dir}")
        else:
            # 从数据集内的 retention_images 加载（兼容旧格式）
            data_dir = Path(forward_file).parent
            retention_dir = data_dir / "retention_images"
            
            cw_path = retention_dir / ("retention_cw_train.jsonl" if split == "train" else "retention_cw_val.jsonl")
            mcq_i2d_path = retention_dir / ("retention_mcq_i2d_train.jsonl" if split == "train" else "retention_mcq_i2d_val.jsonl")
            mcq_d2i_path = retention_dir / ("retention_mcq_d2i_train.jsonl" if split == "train" else "retention_mcq_d2i_val.jsonl")
        
        # Correct/Wrong
        if cw_path.exists():
            with open(cw_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_cw"
                    self.retention_cw.append(sample)
        
        # MCQ I2D
        if mcq_i2d_path.exists():
            with open(mcq_i2d_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_mcq_i2d"
                    self.retention_mcq_i2d.append(sample)
        
        # MCQ D2I
        if mcq_d2i_path.exists():
            with open(mcq_d2i_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_mcq_d2i"
                    self.retention_mcq_d2i.append(sample)
        
        # 计算需要从题库中抽取的样本数量
        # Token 平衡因子：Forward 输出约 10 tokens（人名），Retention 输出 1 token
        # 按 token 数量平衡而非样本数量，需要更多 Retention 样本来匹配梯度贡献
        TOKEN_BALANCE_FACTOR = 10
        
        n_forward = len(self.forward_samples)
        n_retention_total = int(n_forward * retention_ratio / (1 - retention_ratio) * TOKEN_BALANCE_FACTOR)
        n_per_type = n_retention_total // 3
        
        # 从题库中随机抽取（不重采样，如果题库够大）
        random.seed(seed)
        self.all_samples = self.forward_samples.copy()
        
        def sample_from_pool(pool, n):
            """从题库中随机抽取 n 个样本，题库不够则全用上"""
            if not pool:
                return []
            if len(pool) >= n:
                # 题库足够大，随机抽取不重复
                return random.sample(pool, n)
            else:
                # 题库不够，全部使用（不再重采样扩展）
                shuffled = pool.copy()
                random.shuffle(shuffled)
                return shuffled
        
        # 记录实际使用的数量
        cw_used = sample_from_pool(self.retention_cw, n_per_type)
        mcq_i2d_used = sample_from_pool(self.retention_mcq_i2d, n_per_type)
        mcq_d2i_used = sample_from_pool(self.retention_mcq_d2i, n_per_type)
        
        self.all_samples.extend(cw_used)
        self.all_samples.extend(mcq_i2d_used)
        self.all_samples.extend(mcq_d2i_used)
        
        random.shuffle(self.all_samples)
        
        print(f"MixedForwardDataset v4 (pool strategy, {split}):")
        print(f"  Forward: {len(self.forward_samples)}")
        print(f"  Retention CW: {len(cw_used)}/{len(self.retention_cw)} (pool)")
        print(f"  Retention MCQ I2D: {len(mcq_i2d_used)}/{len(self.retention_mcq_i2d)} (pool)")
        print(f"  Retention MCQ D2I: {len(mcq_d2i_used)}/{len(self.retention_mcq_d2i)} (pool)")
        print(f"  Total: {len(self.all_samples)}")
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        task_type = sample.get("task_type", "forward")
        
        if task_type == "forward":
            return self._process_forward(sample)
        elif task_type == "retention_cw":
            return self._process_retention_cw(sample)
        elif task_type == "retention_mcq_i2d":
            return self._process_retention_mcq_i2d(sample)
        elif task_type == "retention_mcq_d2i":
            return self._process_retention_mcq_d2i(sample)
        else:
            return self._process_forward(sample)
    
    def _process_forward(self, sample):
        """Forward训练：[Image] connector → description"""
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        description = sample.get("description", "")
        
        # 格式：[Image] connector
        question = connector
        answer = description
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_cw(self, sample):
        """Retention CW: description connector [Image], correct or wrong? → Correct/Wrong"""
        image = Image.open(sample["image_path"]).convert("RGB")
        object_name = sample["object_name"]
        connector = sample.get("connector", "is")
        label = sample["label"]
        
        # 格式：description connector [Image], correct or wrong? Only answer Correct or Wrong.
        question = f"{object_name} {connector} this image, correct or wrong? Only answer Correct or Wrong."
        answer = label
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_mcq_i2d(self, sample):
        """Retention MCQ I2D: [Image] connector? A. desc1 B. desc2... → A/B/C/D"""
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        choices = sample["choices"]
        correct_index = sample["correct_index"]
        
        # 格式：[Image] connector? A. xxx B. xxx C. xxx D. xxx Only answer A, B, C, or D.
        question = f"{connector}? A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]} Only answer A, B, C, or D."
        answer = chr(65 + correct_index)
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_mcq_d2i(self, sample):
        """Retention MCQ D2I: description connector? A. [img1] B. [img2]... → A/B/C/D"""
        description = sample["description"]
        connector = sample.get("connector", "is")
        image_choices = sample["image_choices"]
        correct_index = sample["correct_index"]
        
        # 加载所有图片
        images = [Image.open(p).convert("RGB") for p in image_choices]
        
        # 格式：description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D.
        question = f"{description} {connector}?"
        suffix = "Only answer A, B, C, or D."
        answer = chr(65 + correct_index)
        
        return self._encode_mcq_d2i(question, images, suffix, answer)
    
    def _encode_single_image(self, image, question, answer):
        """编码单图样本"""
        messages_full = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        messages_prompt = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result
    
    def _encode_mcq_d2i(self, question, images, suffix, answer):
        """编码 MCQ D2I: description connector? A. [img1] B. [img2]..."""
        # 构建消息：question + A. [img] B. [img] C. [img] D. [img] + suffix
        content = [{"type": "text", "text": question + " "}]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"A. " if i == 0 else f" {chr(65+i)}. "})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f" {suffix}"})
        
        messages_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer}
        ]
        
        messages_prompt = [
            {"role": "user", "content": content}
        ]
        
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_full],
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length * 4  # D2I 有4张图，需更大空间
        )
        
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result


def collate_fn(batch):
    """
    DataLoader 的 collate 函数
    
    关键：Qwen3-VL 处理多图时，image_grid_thw 和 pixel_values 需要 concat 而非 stack
    - image_grid_thw: [N, 3] 其中 N = batch 内所有图片总数
    - pixel_values: [total_patches, C] 或类似，沿第0维 concat
    - input_ids/attention_mask/labels: 需要 pad 到相同长度
    """
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        tensors = [item[key] for item in batch]
        
        if key == "pixel_values":
            # pixel_values: 沿第0维 concat（不同样本可能有不同数量的 patches）
            result[key] = torch.cat(tensors, dim=0)
            
        elif key == "image_grid_thw":
            # image_grid_thw: 每个样本是 [num_images, 3]，需要 concat 成 [total_images, 3]
            # 先确保都是 2D
            normalized = []
            for t in tensors:
                if t.dim() == 1:
                    normalized.append(t.unsqueeze(0))  # [3] -> [1, 3]
                else:
                    normalized.append(t)  # 已经是 [N, 3]
            result[key] = torch.cat(normalized, dim=0)  # [total_images, 3]
            
        elif key in ("input_ids", "attention_mask", "labels"):
            # 序列类：需要 pad 到相同长度
            max_len = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad_value = 0 if key == "attention_mask" else -100 if key == "labels" else 0
                    padding = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
                    padded.append(torch.cat([t, padding]))
                else:
                    padded.append(t)
            result[key] = torch.stack(padded)
            
        else:
            # 其他字段尝试直接 stack
            try:
                result[key] = torch.stack(tensors)
            except:
                # 如果 stack 失败，尝试 concat
                result[key] = torch.cat(tensors, dim=0)
    
    return result
"""
==============================================================================
混合数据集 - 用于防止灾难性遗忘的 Forward + 保持任务数据集 (v3)
==============================================================================

训练格式：
  1. Forward训练：[Image] connector → description
  2. Retention CW: description connector [Image], correct or wrong? Only answer Correct or Wrong. → Correct/Wrong
  3. Retention MCQ I2D: [Image] connector? A. desc1 B. desc2 C. desc3 D. desc4 Only answer A, B, C, or D. → A/B/C/D
  4. Retention MCQ D2I: description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D. → A/B/C/D

任务混合：
  - 70% Forward 样本
  - 10% Retention CW
  - 10% Retention MCQ I2D
  - 10% Retention MCQ D2I

==============================================================================
"""

import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class MixedForwardDataset(Dataset):
    """混合 Forward 数据集 (v3 - 统一格式)"""
    
    def __init__(
        self,
        forward_file: str,
        processor,
        max_length: int = 512,
        retention_ratio: float = 0.3,
        seed: int = 42,
        split: str = "train",  # "train" or "val"
        retention_pool_dir: str = None  # 公共题库目录，如 "data/retention_pool"
    ):
        self.processor = processor
        self.max_length = max_length
        self.retention_ratio = retention_ratio
        self.split = split
        
        # 加载 Forward 样本
        self.forward_samples = []
        with open(forward_file) as f:
            for line in f:
                sample = json.loads(line)
                sample["task_type"] = "forward"
                self.forward_samples.append(sample)
        
        # 加载 3 种 Retention 样本
        # 优先使用公共题库（retention_pool_dir），否则使用数据集内的 retention_images
        self.retention_cw = []
        self.retention_mcq_i2d = []
        self.retention_mcq_d2i = []
        
        if retention_pool_dir and Path(retention_pool_dir).exists():
            # 从公共题库加载
            pool_dir = Path(retention_pool_dir)
            suffix = "_train.jsonl" if split == "train" else "_val.jsonl"
            
            cw_path = pool_dir / f"cw{suffix}"
            mcq_i2d_path = pool_dir / f"mcq_i2d{suffix}"
            mcq_d2i_path = pool_dir / f"mcq_d2i{suffix}"
            
            print(f"  Loading retention from pool: {pool_dir}")
        else:
            # 从数据集内的 retention_images 加载（兼容旧格式）
            data_dir = Path(forward_file).parent
            retention_dir = data_dir / "retention_images"
            
            cw_path = retention_dir / ("retention_cw_train.jsonl" if split == "train" else "retention_cw_val.jsonl")
            mcq_i2d_path = retention_dir / ("retention_mcq_i2d_train.jsonl" if split == "train" else "retention_mcq_i2d_val.jsonl")
            mcq_d2i_path = retention_dir / ("retention_mcq_d2i_train.jsonl" if split == "train" else "retention_mcq_d2i_val.jsonl")
        
        # Correct/Wrong
        if cw_path.exists():
            with open(cw_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_cw"
                    self.retention_cw.append(sample)
        
        # MCQ I2D
        if mcq_i2d_path.exists():
            with open(mcq_i2d_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_mcq_i2d"
                    self.retention_mcq_i2d.append(sample)
        
        # MCQ D2I
        if mcq_d2i_path.exists():
            with open(mcq_d2i_path) as f:
                for line in f:
                    sample = json.loads(line)
                    sample["task_type"] = "retention_mcq_d2i"
                    self.retention_mcq_d2i.append(sample)
        
        # 计算需要从题库中抽取的样本数量
        n_forward = len(self.forward_samples)
        n_retention_total = int(n_forward * retention_ratio / (1 - retention_ratio))
        n_per_type = n_retention_total // 3
        
        # 从题库中随机抽取（不重采样，如果题库够大）
        random.seed(seed)
        self.all_samples = self.forward_samples.copy()
        
        def sample_from_pool(pool, n):
            """从题库中随机抽取 n 个样本，题库不够则全用上"""
            if not pool:
                return []
            if len(pool) >= n:
                # 题库足够大，随机抽取不重复
                return random.sample(pool, n)
            else:
                # 题库不够，全部使用（不再重采样扩展）
                shuffled = pool.copy()
                random.shuffle(shuffled)
                return shuffled
        
        # 记录实际使用的数量
        cw_used = sample_from_pool(self.retention_cw, n_per_type)
        mcq_i2d_used = sample_from_pool(self.retention_mcq_i2d, n_per_type)
        mcq_d2i_used = sample_from_pool(self.retention_mcq_d2i, n_per_type)
        
        self.all_samples.extend(cw_used)
        self.all_samples.extend(mcq_i2d_used)
        self.all_samples.extend(mcq_d2i_used)
        
        random.shuffle(self.all_samples)
        
        print(f"MixedForwardDataset v4 (pool strategy, {split}):")
        print(f"  Forward: {len(self.forward_samples)}")
        print(f"  Retention CW: {len(cw_used)}/{len(self.retention_cw)} (pool)")
        print(f"  Retention MCQ I2D: {len(mcq_i2d_used)}/{len(self.retention_mcq_i2d)} (pool)")
        print(f"  Retention MCQ D2I: {len(mcq_d2i_used)}/{len(self.retention_mcq_d2i)} (pool)")
        print(f"  Total: {len(self.all_samples)}")
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        task_type = sample.get("task_type", "forward")
        
        if task_type == "forward":
            return self._process_forward(sample)
        elif task_type == "retention_cw":
            return self._process_retention_cw(sample)
        elif task_type == "retention_mcq_i2d":
            return self._process_retention_mcq_i2d(sample)
        elif task_type == "retention_mcq_d2i":
            return self._process_retention_mcq_d2i(sample)
        else:
            return self._process_forward(sample)
    
    def _process_forward(self, sample):
        """Forward训练：[Image] connector → description"""
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        description = sample.get("description", "")
        
        # 格式：[Image] connector
        question = connector
        answer = description
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_cw(self, sample):
        """Retention CW: description connector [Image], correct or wrong? → Correct/Wrong"""
        image = Image.open(sample["image_path"]).convert("RGB")
        object_name = sample["object_name"]
        connector = sample.get("connector", "is")
        label = sample["label"]
        
        # 格式：description connector [Image], correct or wrong? Only answer Correct or Wrong.
        question = f"{object_name} {connector} this image, correct or wrong? Only answer Correct or Wrong."
        answer = label
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_mcq_i2d(self, sample):
        """Retention MCQ I2D: [Image] connector? A. desc1 B. desc2... → A/B/C/D"""
        image = Image.open(sample["image_path"]).convert("RGB")
        connector = sample.get("connector", "is")
        choices = sample["choices"]
        correct_index = sample["correct_index"]
        
        # 格式：[Image] connector? A. xxx B. xxx C. xxx D. xxx Only answer A, B, C, or D.
        question = f"{connector}? A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]} Only answer A, B, C, or D."
        answer = chr(65 + correct_index)
        
        return self._encode_single_image(image, question, answer)
    
    def _process_retention_mcq_d2i(self, sample):
        """Retention MCQ D2I: description connector? A. [img1] B. [img2]... → A/B/C/D"""
        description = sample["description"]
        connector = sample.get("connector", "is")
        image_choices = sample["image_choices"]
        correct_index = sample["correct_index"]
        
        # 加载所有图片
        images = [Image.open(p).convert("RGB") for p in image_choices]
        
        # 格式：description connector? A. [img1] B. [img2] C. [img3] D. [img4] Only answer A, B, C, or D.
        question = f"{description} {connector}?"
        suffix = "Only answer A, B, C, or D."
        answer = chr(65 + correct_index)
        
        return self._encode_mcq_d2i(question, images, suffix, answer)
    
    def _encode_single_image(self, image, question, answer):
        """编码单图样本"""
        messages_full = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": answer}
        ]
        
        messages_prompt = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}
        ]
        
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result
    
    def _encode_mcq_d2i(self, question, images, suffix, answer):
        """编码 MCQ D2I: description connector? A. [img1] B. [img2]..."""
        # 构建消息：question + A. [img] B. [img] C. [img] D. [img] + suffix
        content = [{"type": "text", "text": question + " "}]
        for i, img in enumerate(images):
            content.append({"type": "text", "text": f"A. " if i == 0 else f" {chr(65+i)}. "})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f" {suffix}"})
        
        messages_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer}
        ]
        
        messages_prompt = [
            {"role": "user", "content": content}
        ]
        
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text_full],
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length * 4  # D2I 有4张图，需更大空间
        )
        
        text_prompt = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        inputs_prompt = self.processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt"
        )
        prompt_len = inputs_prompt["input_ids"].shape[1]
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values_videos"))
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result


