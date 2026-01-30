# -*- coding: utf-8 -*-
"""Quick syntax check for the fixed code"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # 尝试导入修复后的模块
    from experiments import unified_reviewer_experiments as urex
    print("[SUCCESS] Module import successful")

    # 检查关键类和方法
    print(f"\n[CHECK] AblationConfiguration has epsilon: {hasattr(urex.AblationConfiguration, '__dataclass_fields__')}")

    # 获取所有字段
    fields = [f.name for f in urex.AblationConfiguration.__dataclass_fields__.values()]
    print(f"[CHECK] AblationConfiguration fields: {fields}")

    # 创建测试配置
    test_cfg = urex.AblationConfiguration(
        name="test",
        backbone_type="mamba",
        epsilon=0.05
    )
    print(f"[SUCCESS] Created config with epsilon={test_cfg.epsilon}")

    # 检查UnifiedExperimentConfig
    uni_cfg = urex.UnifiedExperimentConfig()
    print(f"[CHECK] UnifiedExperimentConfig n_epochs: {uni_cfg.n_epochs}")

except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[COMPLETE] All checks passed!")
