from typing import Dict, Type
from .BaseLoader import BaseLoader

class LoaderFactory:
    """
    加载器工厂类，负责创建不同格式的加载器实例
    """
    
    # 类变量，存储文件扩展名与加载器类的映射
    _loaders: Dict[str, Type[BaseLoader]] = {}
    
    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseLoader]) -> None:
        """
        注册加载器
        
        Args:
            extension: 文件扩展名（如 '.ply'）
            loader_class: 加载器类
        """
        cls._loaders[extension.lower()] = loader_class
    
    @classmethod
    def create_loader(cls, file_path: str) -> BaseLoader:
        """
        根据文件路径创建合适的加载器实例
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载器实例
            
        Raises:
            ValueError: 不支持的文件格式
        """
        import os
        
        # 获取文件扩展名
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()
        
        # 查找对应的加载器
        loader_class = cls._loaders.get(extension)
        
        if loader_class is None:
            supported = ', '.join(cls._loaders.keys())
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: {supported}")
        
        # 创建并返回加载器实例
        return loader_class()
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        获取支持的文件扩展名列表
        
        Returns:
            支持的文件扩展名列表
        """
        return list(cls._loaders.keys())
