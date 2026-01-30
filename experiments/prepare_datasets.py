# -*- coding: utf-8 -*-
"""
Dataset preparation and caching module for PGMNO experiments.
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.io import loadmat
import json

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 数据集配置
# ============================================================================

DATASET_CONFIG = {
    "burgers_ablation": {
        "n_train": 100,
        "n_val": 20, 
        "n_test": 20,
        "n_grid": 128,
        "nt": 100,
        "dt": 0.01,
        "nu": 0.01 / 3.14159,
        "description": "Burgers数据集用于消融实验"
    },
    "burgers_sensitivity": {
        "n_train": 80,
        "n_val": 20,
        "n_test": 20,
        "n_grid": 128,
        "nt": 100,
        "dt": 0.01,
        "nu": 0.01 / 3.14159,
        "description": "Burgers数据集用于敏感性分析"
    },
    "burgers_physics": {
        "n_train": 0,
        "n_val": 0,
        "n_test": 50,
        "n_grid": 128,
        "nt": 50,
        "dt": 0.01,
        "nu": 0.01 / 3.14159,
        "description": "Burgers数据集用于物理指标分析"
    }
}


# ============================================================================
# 数据目录管理
# ============================================================================

class DataPathManager:
    """数据路径管理器"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # 默认在PGMNO目录下创建data文件夹
            self.base_dir = Path(__file__).parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 现有数据路径
        self.existing_data_paths = {
            "burgers_mat": Path(__file__).parent.parent / "Code" / "gen" / "burgers" / "MgFNO" / "data" / "burgers1d" / "burgers1d_data.mat",
            "rb_data": Path(__file__).parent.parent / "Code" / "gen" / "rb" / "data",
            "sw_mat": Path(__file__).parent.parent / "Code" / "shallow_water_data.mat",
        }
    
    def get_cache_path(self, dataset_name: str) -> Path:
        """获取缓存数据的路径"""
        return self.base_dir / f"{dataset_name}.pt"
    
    def get_config_path(self) -> Path:
        """获取数据配置文件路径"""
        return self.base_dir / "dataset_config.json"
    
    def check_existing_data(self) -> Dict[str, bool]:
        """检查现有数据是否存在"""
        result = {}
        for name, path in self.existing_data_paths.items():
            result[name] = path.exists()
        return result


# ============================================================================
# 数据生成器
# ============================================================================

class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, path_manager: DataPathManager):
        self.path_manager = path_manager
        
    def generate_burgers_dataset(
        self, 
        config_name: str,
        seed: int = 42,
        force_regenerate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        生成Burgers数据集
        
        优先尝试加载现有的.mat数据，否则新生成
        """
        cache_path = self.path_manager.get_cache_path(config_name)
        
        # 如果缓存存在且不强制重新生成
        if cache_path.exists() and not force_regenerate:
            print(f"[Data] Loading cached dataset from {cache_path}")
            return torch.load(cache_path)
        
        config = DATASET_CONFIG[config_name]
        print(f"[Data] Generating {config_name} dataset...")
        print(f"       Config: n_grid={config['n_grid']}, nt={config['nt']}")
        
        # 尝试从现有.mat文件加载
        mat_path = self.path_manager.existing_data_paths.get("burgers_mat")
        if mat_path and mat_path.exists():
            try:
                mat_data = self._load_burgers_from_mat(mat_path, config)
                if mat_data is not None:
                    # 保存缓存
                    torch.save(mat_data, cache_path)
                    print(f"[Data] Cached to {cache_path}")
                    return mat_data
            except Exception as e:
                print(f"[Data] Failed to load from .mat: {e}, generating new data...")
        
        # 新生成数据
        from improved_burgers_utils import generate_burgers_data_v2, create_validation_dataset
        
        dataset = {}
        
        # 训练数据
        if config["n_train"] > 0:
            train_data, x_grid = generate_burgers_data_v2(
                n_samples=config["n_train"],
                n_grid=config["n_grid"],
                nt=config["nt"],
                dt=config["dt"],
                nu=config["nu"],
                seed=seed
            )
            dataset["train"] = train_data
            dataset["x_grid"] = x_grid
            print(f"       Train: {train_data.shape}")
        
        # 验证数据
        if config["n_val"] > 0:
            val_data, _ = create_validation_dataset(
                n_val=config["n_val"],
                n_grid=config["n_grid"],
                nt=config["nt"],
                dt=config["dt"],
                nu=config["nu"],
                seed=seed + 1000
            )
            dataset["val"] = val_data
            print(f"       Val: {val_data.shape}")
        
        # 测试数据
        if config["n_test"] > 0:
            test_data, _ = create_validation_dataset(
                n_val=config["n_test"],
                n_grid=config["n_grid"],
                nt=config["nt"],
                dt=config["dt"],
                nu=config["nu"],
                seed=seed + 2000
            )
            dataset["test"] = test_data
            print(f"       Test: {test_data.shape}")
        
        # 保存配置信息
        dataset["config"] = config
        
        # 缓存
        torch.save(dataset, cache_path)
        print(f"[Data] Cached to {cache_path}")
        
        return dataset
    
    def _load_burgers_from_mat(
        self, 
        mat_path: Path, 
        config: Dict
    ) -> Optional[Dict[str, torch.Tensor]]:
        """从.mat文件加载Burgers数据"""
        print(f"[Data] Trying to load from {mat_path}...")
        
        try:
            mat = loadmat(str(mat_path))
            
            # 尝试常见的字段名
            possible_keys = ['u', 'output', 'data', 'sol', 'solution']
            data = None
            for key in possible_keys:
                if key in mat:
                    data = mat[key]
                    print(f"       Found data under key '{key}' with shape {data.shape}")
                    break
            
            if data is None:
                print(f"       Available keys: {[k for k in mat.keys() if not k.startswith('__')]}")
                return None
            
            # 转换为tensor
            data_tensor = torch.tensor(data, dtype=torch.float32)
            
            # 检查形状是否兼容
            n_samples = config["n_train"] + config["n_val"] + config["n_test"]
            if data_tensor.shape[0] < n_samples:
                print(f"       Insufficient samples: {data_tensor.shape[0]} < {n_samples}")
                return None
            
            # 划分数据集
            dataset = {}
            idx = 0
            
            if config["n_train"] > 0:
                dataset["train"] = data_tensor[idx:idx+config["n_train"]]
                idx += config["n_train"]
            
            if config["n_val"] > 0:
                dataset["val"] = data_tensor[idx:idx+config["n_val"]]
                idx += config["n_val"]
            
            if config["n_test"] > 0:
                dataset["test"] = data_tensor[idx:idx+config["n_test"]]
            
            # 创建x_grid
            n_grid = data_tensor.shape[-1] if len(data_tensor.shape) > 1 else config["n_grid"]
            dataset["x_grid"] = torch.linspace(-1, 1, n_grid)
            dataset["config"] = config
            dataset["source"] = str(mat_path)
            
            print(f"       Successfully loaded from .mat file")
            return dataset
            
        except Exception as e:
            print(f"       Error loading .mat: {e}")
            return None
    
    def prepare_all_datasets(self, force_regenerate: bool = False) -> Dict[str, Path]:
        """准备所有数据集"""
        results = {}
        
        for name in DATASET_CONFIG.keys():
            print(f"\n{'='*50}")
            print(f"Preparing: {name}")
            print(f"{'='*50}")
            
            try:
                self.generate_burgers_dataset(name, force_regenerate=force_regenerate)
                results[name] = self.path_manager.get_cache_path(name)
            except Exception as e:
                print(f"[ERROR] Failed to prepare {name}: {e}")
                results[name] = None
        
        return results


# ============================================================================
# 数据加载器
# ============================================================================

class CachedDataLoader:
    """缓存数据加载器 - 供实验脚本使用"""
    
    def __init__(self, data_dir: str = None):
        self.path_manager = DataPathManager(data_dir)
        self._cache = {}
    
    def load_dataset(self, name: str) -> Dict[str, torch.Tensor]:
        """加载数据集"""
        if name in self._cache:
            return self._cache[name]
        
        cache_path = self.path_manager.get_cache_path(name)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found at {cache_path}. "
                f"Please run 'python prepare_datasets.py --all' first."
            )
        
        dataset = torch.load(cache_path)
        self._cache[name] = dataset
        return dataset
    
    def get_train_val_data(
        self, 
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取训练和验证数据"""
        dataset = self.load_dataset(name)
        return dataset.get("train"), dataset.get("val"), dataset.get("x_grid")
    
    def list_available(self) -> list:
        """列出可用数据集"""
        available = []
        for name in DATASET_CONFIG.keys():
            cache_path = self.path_manager.get_cache_path(name)
            if cache_path.exists():
                available.append(name)
        return available


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PGMNO Dataset Preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare all datasets
    python prepare_datasets.py --all
    
    # List available datasets
    python prepare_datasets.py --list
    
    # Force regenerate
    python prepare_datasets.py --all --force
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Prepare all datasets')
    parser.add_argument('--burgers', action='store_true', help='Prepare Burgers datasets only')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--force', action='store_true', help='Force regenerate even if cached')
    parser.add_argument('--check', action='store_true', help='Check existing data sources')
    
    args = parser.parse_args()
    
    path_manager = DataPathManager()
    generator = DatasetGenerator(path_manager)
    loader = CachedDataLoader()
    
    if args.check:
        print("\n=== Checking Existing Data Sources ===")
        status = path_manager.check_existing_data()
        for name, exists in status.items():
            path = path_manager.existing_data_paths[name]
            status_str = "✓ Found" if exists else "✗ Missing"
            print(f"  {status_str}: {name}")
            print(f"           Path: {path}")
        return
    
    if args.list:
        print("\n=== Available Cached Datasets ===")
        available = loader.list_available()
        if available:
            for name in available:
                cache_path = path_manager.get_cache_path(name)
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {size_mb:.2f} MB")
        else:
            print("  No cached datasets found.")
            print("  Run 'python prepare_datasets.py --all' to generate.")
        
        print("\n=== Dataset Configurations ===")
        for name, config in DATASET_CONFIG.items():
            print(f"  {name}:")
            print(f"    Description: {config['description']}")
            print(f"    Samples: train={config['n_train']}, val={config['n_val']}, test={config['n_test']}")
        return
    
    if args.all or args.burgers:
        print("\n" + "="*60)
        print("PGMNO Dataset Preparation")
        print("="*60)
        
        results = generator.prepare_all_datasets(force_regenerate=args.force)
        
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        for name, path in results.items():
            if path:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"  ✗ {name}: Failed")
        
        print(f"\nData cached in: {path_manager.base_dir}")
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
