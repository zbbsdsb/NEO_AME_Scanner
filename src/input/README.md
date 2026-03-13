# PLYLoader 模块说明

## 功能描述
PLYLoader 是 NEO_AME_Scanner 项目的输入模块，负责读取 ASCII 格式的 PLY 文件，提取 x y z 坐标，并忽略其他未知字段。

## 主要功能
- 支持 ASCII 格式的 PLY 文件读取
- 提取顶点的 x y z 坐标
- 忽略文件中的未知字段
- 提供样本数量统计
- 完善的错误处理机制

## 用法示例

```python
from PLYLoader import PLYLoader

# 创建PLYLoader实例
loader = PLYLoader()

# 加载PLY文件
try:
    vertices = loader.load("path/to/file.ply")
    print(f"成功加载PLY文件，顶点数量: {loader.get_sample_count()}")
    print(f"前3个顶点坐标: {vertices[:3]}")
except Exception as e:
    print(f"加载失败: {str(e)}")

# 获取加载的顶点数据
if loader.get_sample_count() > 0:
    all_vertices = loader.get_vertices()
    print(f"所有顶点数量: {len(all_vertices)}")
```

## 方法说明

### `load(file_path)`
- **参数**：`file_path` - PLY 文件路径
- **返回值**：顶点坐标列表，每个元素为 (x, y, z) 元组
- **异常**：
  - `FileNotFoundError` - 文件不存在
  - `ValueError` - 文件格式错误

### `get_sample_count()`
- **返回值**：加载的样本数量

### `get_vertices()`
- **返回值**：加载的顶点坐标列表

## 测试情况

已编写以下测试用例：
1. 简单 PLY 文件读取
2. 包含额外字段的 PLY 文件读取
3. 文件不存在的情况
4. 格式错误的文件
5. 缺少顶点信息的文件
6. 顶点数据不足的情况

所有测试用例均已通过。

## 技术实现

- 使用 Python 标准库的 `open()` 函数读取文件
- 逐行解析 PLY 文件头部和数据
- 使用 try-except 结构处理异常
- 返回标准的 Python 列表存储顶点坐标

## 注意事项

- 只支持 ASCII 格式的 PLY 文件
- 只读取 x y z 坐标，忽略其他字段
- 对于大型 PLY 文件，可能会占用较多内存
