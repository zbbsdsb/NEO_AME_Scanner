from LoaderFactory import LoaderFactory

# 示例1：使用工厂模式加载PLY文件
try:
    # 工厂根据文件扩展名自动选择合适的加载器
    loader = LoaderFactory.create_loader("sample.ply")
    vertices = loader.load("sample.ply")
    print(f"成功加载PLY文件，顶点数量: {loader.get_sample_count()}")
    print(f"前3个顶点坐标: {vertices[:3]}")
except Exception as e:
    print(f"加载失败: {str(e)}")

# 示例2：获取加载的顶点数据
if loader.get_sample_count() > 0:
    all_vertices = loader.get_vertices()
    print(f"所有顶点数量: {len(all_vertices)}")

# 示例3：查看支持的文件格式
print(f"\n支持的文件格式: {LoaderFactory.get_supported_extensions()}")

# 示例4：加载不同格式的文件（未来扩展）
print("\n尝试加载不同格式的文件：")
file_paths = ["model.ply", "scene.splat", "mesh.obj"]

for file_path in file_paths:
    try:
        loader = LoaderFactory.create_loader(file_path)
        vertices = loader.load(file_path)
        print(f"成功加载 {file_path}，顶点数量: {loader.get_sample_count()}")
    except Exception as e:
        print(f"加载 {file_path} 失败: {str(e)}")
