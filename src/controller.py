# src/controller.py
import pynvml
import logging
from typing import Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPUController")

class GPUController:
    """
    硬件控制层：负责通过 NVML 调整 GPU 的物理参数（如功率上限、频率）。
    对应论文章节: 4.3.3 动态 GPU 频率/功率调节机制
    """
    def __init__(self):
        self._initialized = False
        try:
            pynvml.nvmlInit()
            self._initialized = True
            logger.info("NVML Initialized successfully.")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")

    def get_power_info(self, device_id: int) -> Dict:
        """获取指定 GPU 的当前功率状态和允许范围"""
        if not self._initialized:
            return {}

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            # 获取当前功率 (mW -> W)
            current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            # 获取当前限制 (mW -> W)
            current_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            # 获取允许的调整范围 (mW -> W)
            min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            
            return {
                "device_id": device_id,
                "current_power_w": current_power,
                "current_limit_w": current_limit,
                "min_limit_w": min_limit / 1000.0,
                "max_limit_w": max_limit / 1000.0
            }
        except pynvml.NVMLError as e:
            logger.error(f"Failed to get power info for GPU {device_id}: {e}")
            return {}

    def set_power_limit(self, device_id: int, target_watts: int) -> bool:
        """
        设置 GPU 功率上限 (Power Capping)。
        用于在 Memory Bound 场景下降低功耗。
        """
        if not self._initialized:
            return False

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            # 转换为毫瓦
            target_mw = int(target_watts * 1000)
            
            # 边界检查
            constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            min_limit, max_limit = constraints
            
            # 钳位到安全范围
            safe_target = max(min_limit, min(target_mw, max_limit))
            
            if safe_target != target_mw:
                logger.warning(f"Target {target_watts}W out of bounds. Clamped to {safe_target/1000}W")

            pynvml.nvmlDeviceSetPowerManagementLimit(handle, safe_target)
            logger.info(f"Set GPU {device_id} power limit to {safe_target/1000} W")
            return True
            
        except pynvml.NVMLError_NoPermission:
            logger.error(f"Permission denied setting power for GPU {device_id}. Need sudo/root?")
            return False
        except pynvml.NVMLError as e:
            logger.error(f"Failed to set power for GPU {device_id}: {e}")
            return False

    def reset_power_limit(self, device_id: int) -> bool:
        """重置为最大功率 (恢复默认性能模式)"""
        if not self._initialized:
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            _, max_limit = pynvml.nvmlDeviceGetPowerMaSnagementLimitConstraints(handle)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, max_limit)
            logger.info(f"Reset GPU {device_id} to max power {max_limit/1000} W")
            return True
        except Exception as e:
            logger.error(f"Failed to reset power: {e}")
            return False

    def shutdown(self):
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass