from LoaderFactory import LoaderFactory
from models import Sample, SampleCollection

# 示例1：使用工厂模式加载文件
print("示例1：使用工厂模式加载文件")
try:
    # 工厂根据文件扩展名自动选择合适的加载器
    loader = LoaderFactory.create_loader("sample.ply")
    vertices = loader.load("sample.ply")
    print(f"成功加载文件，顶点数量: {loader.get_sample_count()}")
    print(f"前3个顶点坐标: {vertices[:3]}")
except Exception as e:
    print(f"加载失败: {str(e)}")

# 示例2：查看支持的文件格式
print("\n示例2：查看支持的文件格式")
supported = LoaderFactory.get_supported_extensions()
print(f"支持的文件格式: {supported}")

# 示例3：加载不同格式的文件（未来扩展）
print("\n示例3：加载不同格式的文件")
file_paths = ["model.ply", "scene.splat"]  # 假设未来支持 .splat 格式

for file_path in file_paths:
    try:
        loader = LoaderFactory.create_loader(file_path)
        vertices = loader.load(file_path)
        print(f"成功加载 {file_path}，顶点数量: {loader.get_sample_count()}")
    except Exception as e:
        print(f"加载 {file_path} 失败: {str(e)}")

# 示例4：创建Sample对象
print("\n示例4：创建Sample对象")
sample1 = Sample(1.0, 2.0, 3.0, 0.5)
sample2 = Sample(4.0, 5.0, 6.0)  # 使用默认weight
print(f"Sample 1: {sample1}")
print(f"Sample 2: {sample2}")

# 示例5：从顶点数据创建Sample
print("\n示例5：从顶点数据创建Sample")
vertex = (7.0, 8.0, 9.0)
sample3 = Sample.from_vertex(vertex, 0.8)
print(f"从顶点 {vertex} 创建的Sample: {sample3}")

# 示例6：Sample转换为字典和元组
print("\n示例6：Sample转换为字典和元组")
sample_dict = sample1.to_dict()
sample_tuple = sample1.to_tuple()
print(f"Sample字典: {sample_dict}")
print(f"Sample元组: {sample_tuple}")

# 示例7：使用SampleCollection管理Sample
print("\n示例7：使用SampleCollection管理Sample")
collection = SampleCollection()
collection.add_sample(sample1)
collection.add_sample(sample2)
collection.add_sample(sample3)
print(f"Sample集合大小: {collection.get_sample_count()}")
print(f"Sample集合: {collection}")

# 示例8：从顶点列表创建Sample集合
print("\n示例8：从顶点列表创建Sample集合")
vertices = [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)]
collection2 = SampleCollection()
collection2.add_samples_from_vertices(vertices, 0.6)
print(f"从顶点列表创建的集合大小: {collection2.get_sample_count()}")
print(f"集合中的顶点: {collection2.get_vertices()}")

# 示例9：SampleCollection转换为字典列表
print("\n示例9：SampleCollection转换为字典列表")
sample_list = collection2.to_list()
print(f"字典列表: {sample_list}")

# 示例10：与PLYLoader集成（使用工厂模式）
print("\n示例10：与PLYLoader集成（使用工厂模式）")
try:
    # 模拟PLYLoader的输出
    mock_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    mock_collection = SampleCollection()
    mock_collection.add_samples_from_vertices(mock_vertices)
    print(f"模拟从PLY文件加载的Sample数量: {mock_collection.get_sample_count()}")
    print(f"模拟加载的顶点: {mock_collection.get_vertices()}")
    
except Exception as e:
    print(f"错误: {str(e)}")
