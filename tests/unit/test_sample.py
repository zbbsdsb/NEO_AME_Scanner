import pytest
from src.input.models import Sample, SampleCollection

# 测试用例1：创建Sample对象
def test_sample_creation():
    """
    测试创建Sample对象
    """
    # 创建Sample对象
    sample = Sample(1.0, 2.0, 3.0, 0.5)
    
    # 验证字段值
    assert sample.x == 1.0
    assert sample.y == 2.0
    assert sample.z == 3.0
    assert sample.weight == 0.5

# 测试用例2：使用默认weight创建Sample
def test_sample_with_default_weight():
    """
    测试使用默认weight创建Sample
    """
    # 创建Sample对象，不指定weight
    sample = Sample(1.0, 2.0, 3.0)
    
    # 验证默认weight值
    assert sample.weight == 1.0

# 测试用例3：从顶点数据创建Sample
def test_sample_from_vertex():
    """
    测试从顶点数据创建Sample
    """
    # 顶点数据
    vertex = (1.0, 2.0, 3.0)
    
    # 从顶点创建Sample
    sample = Sample.from_vertex(vertex)
    
    # 验证字段值
    assert sample.x == 1.0
    assert sample.y == 2.0
    assert sample.z == 3.0
    assert sample.weight == 1.0

# 测试用例4：从顶点数据创建Sample并指定weight
def test_sample_from_vertex_with_weight():
    """
    测试从顶点数据创建Sample并指定weight
    """
    # 顶点数据
    vertex = (1.0, 2.0, 3.0)
    
    # 从顶点创建Sample，指定weight
    sample = Sample.from_vertex(vertex, 0.8)
    
    # 验证字段值
    assert sample.x == 1.0
    assert sample.y == 2.0
    assert sample.z == 3.0
    assert sample.weight == 0.8

# 测试用例5：Sample转换为字典
def test_sample_to_dict():
    """
    测试Sample转换为字典
    """
    # 创建Sample对象
    sample = Sample(1.0, 2.0, 3.0, 0.5)
    
    # 转换为字典
    sample_dict = sample.to_dict()
    
    # 验证字典内容
    assert sample_dict['x'] == 1.0
    assert sample_dict['y'] == 2.0
    assert sample_dict['z'] == 3.0
    assert sample_dict['weight'] == 0.5

# 测试用例6：Sample转换为元组
def test_sample_to_tuple():
    """
    测试Sample转换为元组
    """
    # 创建Sample对象
    sample = Sample(1.0, 2.0, 3.0, 0.5)
    
    # 转换为元组
    sample_tuple = sample.to_tuple()
    
    # 验证元组内容
    assert sample_tuple == (1.0, 2.0, 3.0, 0.5)

# 测试用例7：SampleCollection添加Sample
def test_sample_collection_add_sample():
    """
    测试SampleCollection添加Sample
    """
    # 创建SampleCollection
    collection = SampleCollection()
    
    # 创建Sample对象
    sample1 = Sample(1.0, 2.0, 3.0)
    sample2 = Sample(4.0, 5.0, 6.0)
    
    # 添加Sample
    collection.add_sample(sample1)
    collection.add_sample(sample2)
    
    # 验证Sample数量
    assert collection.get_sample_count() == 2

# 测试用例8：SampleCollection从顶点列表添加Sample
def test_sample_collection_add_from_vertices():
    """
    测试SampleCollection从顶点列表添加Sample
    """
    # 创建SampleCollection
    collection = SampleCollection()
    
    # 顶点列表
    vertices = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    
    # 从顶点列表添加Sample
    collection.add_samples_from_vertices(vertices)
    
    # 验证Sample数量
    assert collection.get_sample_count() == 3

# 测试用例9：SampleCollection获取顶点坐标
def test_sample_collection_get_vertices():
    """
    测试SampleCollection获取顶点坐标
    """
    # 创建SampleCollection
    collection = SampleCollection()
    
    # 顶点列表
    vertices = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    
    # 从顶点列表添加Sample
    collection.add_samples_from_vertices(vertices)
    
    # 获取顶点坐标
    retrieved_vertices = collection.get_vertices()
    
    # 验证顶点坐标
    assert retrieved_vertices == vertices

# 测试用例10：SampleCollection转换为字典列表
def test_sample_collection_to_list():
    """
    测试SampleCollection转换为字典列表
    """
    # 创建SampleCollection
    collection = SampleCollection()
    
    # 创建Sample对象
    sample1 = Sample(1.0, 2.0, 3.0)
    sample2 = Sample(4.0, 5.0, 6.0, 0.5)
    
    # 添加Sample
    collection.add_sample(sample1)
    collection.add_sample(sample2)
    
    # 转换为字典列表
    sample_list = collection.to_list()
    
    # 验证列表内容
    assert len(sample_list) == 2
    assert sample_list[0]['x'] == 1.0
    assert sample_list[0]['weight'] == 1.0
    assert sample_list[1]['x'] == 4.0
    assert sample_list[1]['weight'] == 0.5

# 测试用例11：错误处理 - 顶点数据不足
def test_sample_from_vertex_insufficient_data():
    """
    测试顶点数据不足的情况
    """
    # 顶点数据不足（少于3个坐标）
    vertex = (1.0, 2.0)
    
    # 预期抛出ValueError
    with pytest.raises(ValueError):
        Sample.from_vertex(vertex)

# 测试用例12：错误处理 - 添加非Sample对象
def test_sample_collection_add_non_sample():
    """
    测试添加非Sample对象的情况
    """
    # 创建SampleCollection
    collection = SampleCollection()
    
    # 尝试添加非Sample对象
    with pytest.raises(TypeError):
        collection.add_sample("not a sample")
