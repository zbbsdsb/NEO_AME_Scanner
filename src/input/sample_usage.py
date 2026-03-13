from PLYLoader import PLYLoader
from models import Sample, SampleCollection

# 示例1：创建Sample对象
print("示例1：创建Sample对象")
sample1 = Sample(1.0, 2.0, 3.0, 0.5)
sample2 = Sample(4.0, 5.0, 6.0)  # 使用默认weight
print(f"Sample 1: {sample1}")
print(f"Sample 2: {sample2}")
print()

# 示例2：从顶点数据创建Sample
print("示例2：从顶点数据创建Sample")
vertex = (7.0, 8.0, 9.0)
sample3 = Sample.from_vertex(vertex, 0.8)
print(f"从顶点 {vertex} 创建的Sample: {sample3}")
print()

# 示例3：Sample转换为字典和元组
print("示例3：Sample转换为字典和元组")
sample_dict = sample1.to_dict()
sample_tuple = sample1.to_tuple()
print(f"Sample字典: {sample_dict}")
print(f"Sample元组: {sample_tuple}")
print()

# 示例4：使用SampleCollection管理Sample
print("示例4：使用SampleCollection管理Sample")
collection = SampleCollection()
collection.add_sample(sample1)
collection.add_sample(sample2)
collection.add_sample(sample3)
print(f"Sample集合大小: {collection.get_sample_count()}")
print(f"Sample集合: {collection}")
print()

# 示例5：从顶点列表创建Sample集合
print("示例5：从顶点列表创建Sample集合")
vertices = [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)]
collection2 = SampleCollection()
collection2.add_samples_from_vertices(vertices, 0.6)
print(f"从顶点列表创建的集合大小: {collection2.get_sample_count()}")
print(f"集合中的顶点: {collection2.get_vertices()}")
print()

# 示例6：SampleCollection转换为字典列表
print("示例6：SampleCollection转换为字典列表")
sample_list = collection2.to_list()
print(f"字典列表: {sample_list}")
print()

# 示例7：与PLYLoader集成
print("示例7：与PLYLoader集成")
try:
    # 创建PLYLoader实例
    loader = PLYLoader()
    
    # 这里使用一个假设的文件路径，实际使用时需要替换为真实的PLY文件路径
    # vertices = loader.load("sample.ply")
    # 
    # # 从PLYLoader的输出创建Sample集合
    # ply_collection = SampleCollection()
    # ply_collection.add_samples_from_vertices(vertices)
    # 
    # print(f"从PLY文件加载的Sample数量: {ply_collection.get_sample_count()}")
    
    # 模拟PLYLoader的输出
    mock_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    mock_collection = SampleCollection()
    mock_collection.add_samples_from_vertices(mock_vertices)
    print(f"模拟从PLY文件加载的Sample数量: {mock_collection.get_sample_count()}")
    print(f"模拟加载的顶点: {mock_collection.get_vertices()}")
    
except Exception as e:
    print(f"错误: {str(e)}")
