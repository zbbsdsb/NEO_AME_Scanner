class Sample:
    """
    样本数据结构，包含坐标和权重信息
    """
    def __init__(self, x, y, z, weight=1.0):
        """
        初始化Sample对象
        
        Args:
            x: x坐标
            y: y坐标
            z: z坐标
            weight: 权重，默认为1.0
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.weight = float(weight)
    
    def to_dict(self):
        """
        转换为字典格式
        
        Returns:
            dict: 包含Sample字段的字典
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'weight': self.weight
        }
    
    def to_tuple(self):
        """
        转换为元组格式
        
        Returns:
            tuple: (x, y, z, weight)
        """
        return (self.x, self.y, self.z, self.weight)
    
    @classmethod
    def from_vertex(cls, vertex, weight=1.0):
        """
        从顶点数据创建Sample
        
        Args:
            vertex: 顶点坐标元组 (x, y, z)
            weight: 权重，默认为1.0
            
        Returns:
            Sample: Sample对象
        """
        if len(vertex) < 3:
            raise ValueError("顶点数据至少需要3个坐标值")
        x, y, z = vertex[:3]
        return cls(x, y, z, weight)
    
    def __repr__(self):
        """
        字符串表示
        """
        return f"Sample(x={self.x}, y={self.y}, z={self.z}, weight={self.weight})"

class SampleCollection:
    """
    样本集合管理类
    """
    def __init__(self):
        """
        初始化SampleCollection对象
        """
        self.samples = []
    
    def add_sample(self, sample):
        """
        添加Sample到集合
        
        Args:
            sample: Sample对象
        """
        if not isinstance(sample, Sample):
            raise TypeError("只能添加Sample类型的对象")
        self.samples.append(sample)
    
    def add_samples_from_vertices(self, vertices, weight=1.0):
        """
        从顶点列表添加多个Sample
        
        Args:
            vertices: 顶点列表，每个元素为(x, y, z)元组
            weight: 权重，默认为1.0
        """
        for vertex in vertices:
            sample = Sample.from_vertex(vertex, weight)
            self.add_sample(sample)
    
    def get_samples(self):
        """
        获取所有Sample
        
        Returns:
            list: Sample对象列表
        """
        return self.samples
    
    def get_sample_count(self):
        """
        获取Sample数量
        
        Returns:
            int: Sample数量
        """
        return len(self.samples)
    
    def get_vertices(self):
        """
        获取所有顶点坐标
        
        Returns:
            list: 顶点坐标列表，每个元素为(x, y, z)元组
        """
        return [(sample.x, sample.y, sample.z) for sample in self.samples]
    
    def to_list(self):
        """
        转换为字典列表
        
        Returns:
            list: 包含Sample字典的列表
        """
        return [sample.to_dict() for sample in self.samples]
    
    def __repr__(self):
        """
        字符串表示
        """
        return f"SampleCollection(sample_count={self.get_sample_count()})"
