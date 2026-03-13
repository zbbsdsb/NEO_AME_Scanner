import os
import tempfile
import pytest
from src.input.PLYLoader import PLYLoader

# 创建测试用的PLY文件
def create_test_ply_file(content):
    """
    创建临时PLY文件
    
    Args:
        content: PLY文件内容
        
    Returns:
        str: 临时文件路径
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(content)
        return f.name

# 测试用例1：简单PLY文件
def test_simple_ply_file():
    """
    测试简单PLY文件的读取
    """
    # 创建简单的PLY文件内容
    ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 0
property list uchar int vertex_index
end_header
0 0 0
1 0 0
0 1 0
"""
    
    # 创建临时文件
    temp_file = create_test_ply_file(ply_content)
    
    try:
        # 加载文件
        loader = PLYLoader()
        vertices = loader.load(temp_file)
        
        # 验证结果
        assert len(vertices) == 3
        assert loader.get_sample_count() == 3
        assert vertices[0] == (0.0, 0.0, 0.0)
        assert vertices[1] == (1.0, 0.0, 0.0)
        assert vertices[2] == (0.0, 1.0, 0.0)
    finally:
        # 清理临时文件
        os.unlink(temp_file)

# 测试用例2：包含额外字段的PLY文件
def test_ply_file_with_extra_fields():
    """
    测试包含额外字段的PLY文件读取
    """
    # 创建包含额外字段的PLY文件内容
    ply_content = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 255 0 0
1 1 1 0 255 0
"""
    
    # 创建临时文件
    temp_file = create_test_ply_file(ply_content)
    
    try:
        # 加载文件
        loader = PLYLoader()
        vertices = loader.load(temp_file)
        
        # 验证结果（只读取x, y, z坐标）
        assert len(vertices) == 2
        assert loader.get_sample_count() == 2
        assert vertices[0] == (0.0, 0.0, 0.0)
        assert vertices[1] == (1.0, 1.0, 1.0)
    finally:
        # 清理临时文件
        os.unlink(temp_file)

# 测试用例3：文件不存在的情况
def test_nonexistent_file():
    """
    测试文件不存在的情况
    """
    loader = PLYLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_file.ply")

# 测试用例4：格式错误的文件
def test_invalid_ply_format():
    """
    测试格式错误的PLY文件
    """
    # 创建格式错误的文件内容
    invalid_content = """invalid ply file
this is not a valid PLY file
"""
    
    temp_file = create_test_ply_file(invalid_content)
    
    try:
        loader = PLYLoader()
        with pytest.raises(ValueError):
            loader.load(temp_file)
    finally:
        os.unlink(temp_file)

# 测试用例5：缺少顶点信息的文件
def test_missing_vertex_info():
    """
    测试缺少顶点信息的PLY文件
    """
    # 创建缺少顶点信息的文件内容
    content = """ply
format ascii 1.0
end_header
0 0 0
1 0 0
"""
    
    temp_file = create_test_ply_file(content)
    
    try:
        loader = PLYLoader()
        with pytest.raises(ValueError):
            loader.load(temp_file)
    finally:
        os.unlink(temp_file)

# 测试用例6：顶点数据不足的情况
def test_insufficient_vertex_data():
    """
    测试顶点数据不足的情况
    """
    # 创建顶点数据不足的文件内容
    content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 0 0
"""
    
    temp_file = create_test_ply_file(content)
    
    try:
        loader = PLYLoader()
        with pytest.raises(ValueError):
            loader.load(temp_file)
    finally:
        os.unlink(temp_file)
