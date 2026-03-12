# NEO_AME_Scanner - 实现计划

## 项目概述
根据设计文档，NEO_AME_Scanner 是 AME 系统中的 Local Geometric Evidence Extractor，负责从空间重建数据中提取几何存在证据，输出为 fragment 单元。

## 核心设计要点
- **角色定位**：Local Geometric Evidence Extractor
- **输入**：.ply、.splat 等空间重建数据，统一为 SpatialEvidenceField
- **输出**：ScannerOutput 结构，包含 fragments、scan_quality 等
- **算法主线**：Load → Normalize → Density Collapse → Cluster → Fragment Filter → OBB Compute → Output
- **核心原则**：geometry only、fragment-first、minimal but stable

## 开发计划（分解任务）

### [ ] 任务 1：项目初始化与基础架构搭建
- **优先级**：P0
- **依赖**：None
- **描述**：
  - 创建项目目录结构
  - 配置开发环境和依赖管理
  - 建立基础代码框架
- **成功标准**：
  - 项目结构清晰，符合模块化设计
  - 环境配置完成，可正常构建
- **测试要求**：
  - `programmatic` TR-1.1：项目能够正常构建，无编译错误
  - `human-judgement` TR-1.2：目录结构合理，代码组织清晰
- **备注**：选择合适的编程语言（建议 Python 或 C++），考虑 3D 处理库的集成

### [ ] 任务 2：输入处理模块实现
- **优先级**：P0
- **依赖**：任务 1
- **描述**：
  - 实现 .ply 文件读取功能
  - 实现 .splat 文件读取功能
  - 设计并实现 SpatialEvidenceField 数据结构
  - 统一不同输入格式的数据表示
- **成功标准**：
  - 能够正确读取并解析 .ply 和 .splat 文件
  - 生成符合规范的 SpatialEvidenceField 数据
- **测试要求**：
  - `programmatic` TR-2.1：成功读取测试 .ply 和 .splat 文件
  - `programmatic` TR-2.2：SpatialEvidenceField 数据结构正确存储位置、密度等信息
- **备注**：考虑使用现有的 3D 库处理文件格式解析

### [ ] 任务 3：Field Normalization 模块实现
- **优先级**：P1
- **依赖**：任务 2
- **描述**：
  - 实现输入数据的归一化处理
  - 处理不同尺度和单位的输入数据
  - 确保数据分布合理，便于后续处理
- **成功标准**：
  - 输入数据经过归一化后，具有统一的尺度和分布
- **测试要求**：
  - `programmatic` TR-3.1：归一化后的数据范围合理
  - `programmatic` TR-3.2：不同输入格式的数据归一化结果一致

### [ ] 任务 4：Density Collapse 核心算法实现
- **优先级**：P0
- **依赖**：任务 3
- **描述**：
  - 实现 phantom density 移除算法
  - 保留稳定的几何结构
  - 优化密度阈值和过滤策略
- **成功标准**：
  - 能够有效移除噪声和幻影密度
  - 保留真实的几何结构
- **测试要求**：
  - `programmatic` TR-4.1：处理后的数据噪声减少，几何结构清晰
  - `human-judgement` TR-4.2：可视化结果显示几何结构完整
- **备注**：这是 scanner 成败的关键，需要重点优化

### [ ] 任务 5：Spatial Partition 核心算法实现
- **优先级**：P0
- **依赖**：任务 4
- **描述**：
  - 实现 Voxel Connectivity 分析
  - 实现 Cluster Formation（DBSCAN 或 Voxel Region Growing）
  - 实现 Fragment Refinement（小 cluster 合并、极端形状处理等）
- **成功标准**：
  - 能够稳定地将空间证据切分为合理的 fragment
  - 避免 fragment 爆炸和粘连
- **测试要求**：
  - `programmatic` TR-5.1：fragment 数量在合理范围（20-200）
  - `programmatic` TR-5.2：多次运行结果一致，稳定性良好
- **备注**：参考 Fragment Partition.md 文档中的参数建议

### [ ] 任务 6：OBB 计算模块实现
- **优先级**：P1
- **依赖**：任务 5
- **描述**：
  - 为每个 fragment 计算 Oriented Bounding Box
  - 确保 OBB 能够准确包围 fragment 几何
- **成功标准**：
  - 计算的 OBB 能够正确包围 fragment
  - 计算效率合理
- **测试要求**：
  - `programmatic` TR-6.1：OBB 计算结果正确
  - `programmatic` TR-6.2：计算性能满足实时要求

### [ ] 任务 7：输出模块实现
- **优先级**：P1
- **依赖**：任务 6
- **描述**：
  - 实现 ScannerOutput 结构生成
  - 填充 scene_frame、fragments、scan_quality、provenance 等字段
  - 支持输出为标准格式
- **成功标准**：
  - 生成符合设计规范的 ScannerOutput 结构
  - 输出格式正确，便于后续系统使用
- **测试要求**：
  - `programmatic` TR-7.1：输出结构符合设计规范
  - `programmatic` TR-7.2：输出数据可被其他系统正确解析

### [ ] 任务 8：整体 Pipeline 集成
- **优先级**：P0
- **依赖**：任务 2-7
- **描述**：
  - 集成所有模块，构建完整的 Scanner pipeline
  - 实现模块间的数据流转
  - 优化整体性能和稳定性
- **成功标准**：
  - 完整 pipeline 能够端到端处理输入数据
  - 输出结果符合设计要求
- **测试要求**：
  - `programmatic` TR-8.1：完整 pipeline 运行无错误
  - `programmatic` TR-8.2：处理速度满足要求

### [ ] 任务 9：测试与验证
- **优先级**：P1
- **依赖**：任务 8
- **描述**：
  - 编写单元测试和集成测试
  - 使用测试数据验证功能正确性
  - 性能测试和优化
- **成功标准**：
  - 测试覆盖率达到 80% 以上
  - 所有测试用例通过
- **测试要求**：
  - `programmatic` TR-9.1：单元测试通过率 100%
  - `programmatic` TR-9.2：集成测试通过

### [ ] 任务 10：文档与部署
- **优先级**：P2
- **依赖**：任务 9
- **描述**：
  - 编写详细的代码文档
  - 制作使用说明和部署指南
  - 准备示例数据和演示
- **成功标准**：
  - 文档完整，便于使用和维护
  - 部署流程清晰，易于集成到 AME 系统
- **测试要求**：
  - `human-judgement` TR-10.1：文档结构清晰，内容完整
  - `programmatic` TR-10.2：部署流程可执行

## 技术栈建议
- **编程语言**：Python（开发效率高）或 C++（性能优）
- **3D 处理库**：Open3D、PyVista 或 CGAL
- **数据结构**：NumPy、Pandas
- **测试框架**：pytest 或 Google Test
- **构建工具**：CMake（C++）或 setuptools（Python）

## 时间估计
- 任务 1-2：1 周
- 任务 3-4：2 周（Density Collapse 为核心）
- 任务 5：2 周（Spatial Partition 为核心）
- 任务 6-7：1 周
- 任务 8-9：1 周
- 任务 10：1 周

**总估计时间**：8 周

## 风险评估
- **技术风险**：Density Collapse 和 Spatial Partition 算法的实现效果
- **性能风险**：处理大规模 3D 数据时的性能问题
- **集成风险**：与 AME 系统其他组件的接口兼容性

## 应对策略
- 采用模块化设计，便于单独测试和优化
- 预留性能优化时间，必要时采用并行计算
- 与 AME 系统其他组件保持密切沟通，确保接口一致性