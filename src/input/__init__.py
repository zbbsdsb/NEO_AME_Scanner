from .BaseLoader import BaseLoader
from .PLYLoader import PLYLoader
from .LoaderFactory import LoaderFactory

# 注册所有加载器
LoaderFactory.register_loader('.ply', PLYLoader)

# 如果未来添加 SPLATLoader，可以这样注册：
# from SPLATLoader import SPLATLoader
# LoaderFactory.register_loader('.splat', SPLATLoader)

__all__ = ['BaseLoader', 'PLYLoader', 'LoaderFactory']
