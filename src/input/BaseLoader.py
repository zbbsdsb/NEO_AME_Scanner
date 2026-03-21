from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseLoader(ABC):
    """
    加载器抽象基类，定义所有加载器的公共接口
    """
    
    def __init__(self):
        """
        初始化加载器
        """
        self.vertices: List[Tuple[float, float, float]] = []
        self.sample_count: int = 0
        self.file_path: str = None
    
    @abstractmethod
    def load(self, file_path: str) -> List[Tuple[float, float, float]]:
        """
        加载文件并返回顶点数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            顶点坐标列表，每个元素为 (x, y, z) 元组
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass
    
    def get_sample_count(self) -> int:
        """
        获取样本数量
        
        Returns:
            样本数量
        """
        return self.sample_count
    
    def get_vertices(self) -> List[Tuple[float, float, float]]:
        """
        获取顶点数据
        
        Returns:
            顶点坐标列表
        """
        return self.vertices
    
    def get_file_path(self) -> str:
        """
        获取文件路径
        
        Returns:
            文件路径
        """
        return self.file_path
    
    def _validate_file(self, file_path: str) -> bool:
        """
        验证文件是否存在
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件是否存在
        """
        import os
        return os.path.exists(file_path)
