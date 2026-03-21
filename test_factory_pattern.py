"""
测试工厂模式功能的脚本
"""
import os
import tempfile
from src.input.LoaderFactory import LoaderFactory
from src.input.PLYLoader import PLYLoader
from src.input.models import Sample, SampleCollection

def test_factory_pattern():
    """测试工厂模式的基本功能"""
    print("=" * 60)
    print("测试工厂模式功能")
    print("=" * 60)
    
    # 测试1：检查支持的文件格式
    print("\n测试1：检查支持的文件格式")
    supported = LoaderFactory.get_supported_extensions()
    print(f"支持的文件格式: {supported}")
    assert '.ply' in supported, "应该支持 .ply 格式"
    print("✓ 通过")
    
    # 测试2：创建 PLY 加载器
    print("\n测试2：创建 PLY 加载器")
    loader = LoaderFactory.create_loader("test.ply")
    assert isinstance(loader, PLYLoader), "应该创建 PLYLoader 实例"
    print(f"✓ 创建的加载器类型: {type(loader).__name__}")
    
    # 测试3：加载真实的 PLY 文件
    print("\n测试3：加载真实的 PLY 文件")
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(ply_content)
        temp_file = f.name
    
    try:
        loader = LoaderFactory.create_loader(temp_file)
        vertices = loader.load(temp_file)
        
        print(f"✓ 成功加载文件: {temp_file}")
        print(f"  顶点数量: {loader.get_sample_count()}")
        print(f"  顶点数据: {vertices}")
        
        assert loader.get_sample_count() == 3, "应该加载 3 个顶点"
        assert vertices[0] == (0.0, 0.0, 0.0), "第一个顶点应该是 (0, 0, 0)"
        assert loader.get_file_path() == temp_file, "文件路径应该正确"
        print("✓ 通过")
    finally:
        os.unlink(temp_file)
    
    # 测试4：不支持的文件格式
    print("\n测试4：不支持的文件格式")
    try:
        loader = LoaderFactory.create_loader("test.obj")
        print("✗ 应该抛出异常")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"✓ 正确抛出异常: {e}")
    
    # 测试5：与 SampleCollection 集成
    print("\n测试5：与 SampleCollection 集成")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(ply_content)
        temp_file = f.name
    
    try:
        loader = LoaderFactory.create_loader(temp_file)
        vertices = loader.load(temp_file)
        
        collection = SampleCollection()
        collection.add_samples_from_vertices(vertices)
        
        print(f"✓ SampleCollection 创建成功")
        print(f"  样本数量: {collection.get_sample_count()}")
        print(f"  顶点坐标: {collection.get_vertices()}")
        
        assert collection.get_sample_count() == 3, "SampleCollection 应该有 3 个样本"
        print("✓ 通过")
    finally:
        os.unlink(temp_file)
    
    # 测试6：向后兼容性 - 直接创建 PLYLoader
    print("\n测试6：向后兼容性 - 直接创建 PLYLoader")
    direct_loader = PLYLoader()
    print(f"✓ 可以直接创建 PLYLoader 实例: {type(direct_loader).__name__}")
    
    # 验证继承关系
    from src.input.BaseLoader import BaseLoader
    assert isinstance(direct_loader, BaseLoader), "PLYLoader 应该继承 BaseLoader"
    print(f"✓ PLYLoader 继承自 BaseLoader")
    
    # 验证公共方法
    assert hasattr(direct_loader, 'get_sample_count'), "应该有 get_sample_count 方法"
    assert hasattr(direct_loader, 'get_vertices'), "应该有 get_vertices 方法"
    assert hasattr(direct_loader, 'get_file_path'), "应该有 get_file_path 方法"
    print("✓ 所有公共方法都存在")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)

if __name__ == "__main__":
    test_factory_pattern()
