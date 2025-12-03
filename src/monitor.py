"""
PowerMonitor: 封装 pynvml 进行多 GPU 功耗监控
"""
import time
import threading
import pynvml
import numpy as np

class PowerMonitor:
    """
    GPU 功耗监控类，支持多卡实时功耗采集和后台线程采样。
    在多卡模式下，记录的是所有指定 GPU 的【总功率】。
    """
    
    def __init__(self, device_ids, sampling_interval=0.1):
        """
        初始化功耗监控器
        
        Args:
            device_ids: 监控的 GPU ID 列表 (List[int]) 或单个 ID (int)
            sampling_interval: 采样间隔（秒），默认 0.1 秒
        """
        # 兼容处理：如果是单个 int，转为 list
        if isinstance(device_ids, int):
            self.device_ids = [device_ids]
        else:
            self.device_ids = list(device_ids)
            
        self.sampling_interval = sampling_interval
        self.handles = []  # 存储所有受控 GPU 的句柄
        self.samples = []  # 存储 (timestamp, total_power_watts) 元组
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.start_time = None
        self._nvml_initialized = False
    
    def initialize(self):
        """
        初始化 NVML 并获取所有目标设备的句柄
        """
        try:
            if not self._nvml_initialized:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            
            self.handles = []
            for dev_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
                self.handles.append(handle)
            
            print(f"功耗监控器已初始化，监控 GPU 列表: {self.device_ids}")
        except Exception as e:
            print(f"警告: 无法初始化功耗监控: {e}")
            self.handles = []
    
    def _monitor_loop(self):
        """后台监控循环：聚合所有 GPU 的功率"""
        if not self.handles:
            return
        
        while not self.stop_event.is_set():
            try:
                timestamp = time.time()
                total_power_mw = 0.0
                
                # 遍历所有句柄，累加功率
                for handle in self.handles:
                    total_power_mw += pynvml.nvmlDeviceGetPowerUsage(handle)
                
                total_power_watts = total_power_mw / 1000.0  # 转换为瓦特
                self.samples.append((timestamp, total_power_watts))
            except Exception as e:
                # 生产环境为了不刷屏，可以适当降低报错频率
                print(f"警告: 功耗采样失败: {e}")
            
            # 等待采样间隔
            self.stop_event.wait(self.sampling_interval)
    
    def start(self):
        """启动监控线程"""
        if not self.handles:
            # 尝试自动初始化
            self.initialize()
            if not self.handles:
                print("错误: 无法启动监控，未找到有效的 GPU 句柄")
                return
        
        self.start_time = time.time()
        self.samples = []  # 清空旧数据
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"功耗监控线程已启动 (Devices: {self.device_ids})")
    
    def stop(self):
        """停止监控线程"""
        if self.monitor_thread is not None:
            self.stop_event.set()
            self.monitor_thread.join(timeout=2.0)
            print(f"功耗监控线程已停止，共采集 {len(self.samples)} 个聚合样本")
    
    def calculate_energy(self, start_time, end_time, prefill_end_time=None):
        """
        计算指定时间段内的总能耗 (Joules) 和平均功率 (Watts)。
        逻辑保持不变，因为 self.samples 存的已经是总功率了。
        """
        if not self.samples:
            return 0.0, 0.0, 0.0, 0.0
        
        # 转换为 numpy 数组处理更高效
        data = np.array(self.samples)
        timestamps = data[:, 0]
        powers = data[:, 1]
        
        # 1. 截取 [start_time, end_time] 区间
        # 为了容错，如果 start_time 比第一个采样点还早，就从头算起
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        if not np.any(mask):
            return 0.0, 0.0, 0.0, 0.0
        
        t_valid = timestamps[mask]
        p_valid = powers[mask]
        
        if len(t_valid) < 2:
            return 0.0, np.mean(p_valid), 0.0, 0.0
        
        # 使用梯形法则计算积分 (Energy = Power * Time)
        total_energy = np.trapz(p_valid, t_valid)
        total_time_duration = t_valid[-1] - t_valid[0]
        
        avg_power = total_energy / total_time_duration if total_time_duration > 0 else 0.0
        
        # 分离 Prefill 和 Decode (如果有)
        prefill_avg = 0.0
        decode_avg = 0.0
        
        if prefill_end_time:
            # Prefill 区间
            mask_pre = (timestamps >= start_time) & (timestamps <= prefill_end_time)
            if np.any(mask_pre) and len(timestamps[mask_pre]) > 1:
                t_pre = timestamps[mask_pre]
                p_pre = powers[mask_pre]
                e_pre = np.trapz(p_pre, t_pre)
                prefill_avg = e_pre / (t_pre[-1] - t_pre[0])
            else:
                prefill_avg = avg_power # 兜底
            
            # Decode 区间
            mask_dec = (timestamps > prefill_end_time) & (timestamps <= end_time)
            if np.any(mask_dec) and len(timestamps[mask_dec]) > 1:
                t_dec = timestamps[mask_dec]
                p_dec = powers[mask_dec]
                e_dec = np.trapz(p_dec, t_dec)
                decode_avg = e_dec / (t_dec[-1] - t_dec[0])
            else:
                decode_avg = avg_power # 兜底
                
        return total_energy, avg_power, prefill_avg, decode_avg

    def shutdown(self):
        """关闭 NVML"""
        try:
            if self._nvml_initialized:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
        except:
            pass