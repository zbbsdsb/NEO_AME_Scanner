from PLYLoader import PLYLoader

# 创建PLYLoader实例
loader = PLYLoader()

# 示例1：加载简单的PLY文件
try:
    # 这里使用一个假设的文件路径，实际使用时需要替换为真实的PLY文件路径
    vertices = loader.load("sample.ply")
    print(f"成功加载PLY文件，顶点数量: {loader.get_sample_count()}")
    print(f"前3个顶点坐标: {vertices[:3]}")
except Exception as e:
    print(f"加载失败: {str(e)}")

# 示例2：获取加载的顶点数据
if loader.get_sample_count() > 0:
    all_vertices = loader.get_vertices()
    print(f"所有顶点数量: {len(all_vertices)}")
