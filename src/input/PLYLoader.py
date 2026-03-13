class PLYLoader:
    def __init__(self):
        self.vertices = []
        self.sample_count = 0
        self.file_path = None
    
    def load(self, file_path):
        """
        加载PLY文件
        
        Args:
            file_path: PLY文件路径
            
        Returns:
            list: 顶点坐标列表，每个元素为(x, y, z)元组
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        self.file_path = file_path
        self.vertices = []
        
        try:
            with open(file_path, 'r') as f:
                # 解析头部
                header_info = self._parse_header(f)
                vertex_count = header_info['vertex_count']
                
                # 读取顶点数据
                self.vertices = self._read_vertices(f, vertex_count)
                
                # 验证顶点数量
                if len(self.vertices) != vertex_count:
                    raise ValueError(f"顶点数量不匹配: 预期 {vertex_count}, 实际 {len(self.vertices)}")
                
                self.sample_count = len(self.vertices)
                return self.vertices
                
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except Exception as e:
            raise ValueError(f"解析PLY文件失败: {str(e)}")
    
    def _parse_header(self, file):
        """
        解析PLY文件头部
        
        Args:
            file: 文件对象
            
        Returns:
            dict: 头部信息，包含vertex_count等
        """
        header_info = {}
        
        # 读取第一行，验证文件类型
        first_line = file.readline().strip()
        if first_line != 'ply':
            raise ValueError("不是有效的PLY文件")
        
        # 读取格式信息
        format_line = file.readline().strip()
        if not format_line.startswith('format ascii'):
            raise ValueError("只支持ASCII格式的PLY文件")
        
        # 读取头部其他信息
        while True:
            line = file.readline().strip()
            if line == 'end_header':
                break
            
            if line.startswith('element vertex'):
                # 提取顶点数量
                parts = line.split()
                if len(parts) >= 3:
                    header_info['vertex_count'] = int(parts[2])
            
            # 忽略其他头部信息
        
        if 'vertex_count' not in header_info:
            raise ValueError("PLY文件头部缺少顶点信息")
        
        return header_info
    
    def _read_vertices(self, file, vertex_count):
        """
        读取顶点数据
        
        Args:
            file: 文件对象
            vertex_count: 顶点数量
            
        Returns:
            list: 顶点坐标列表
        """
        vertices = []
        
        for i in range(vertex_count):
            line = file.readline().strip()
            if not line:
                raise ValueError(f"顶点数据不足，预期 {vertex_count} 个顶点")
            
            # 分割行数据
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"顶点数据格式错误，至少需要3个坐标值")
            
            try:
                # 只读取前三个值作为x, y, z坐标
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                vertices.append((x, y, z))
            except ValueError:
                raise ValueError(f"顶点坐标值不是有效的数字")
        
        return vertices
    
    def get_sample_count(self):
        """
        获取样本数量
        
        Returns:
            int: 样本数量
        """
        return self.sample_count
    
    def get_vertices(self):
        """
        获取加载的顶点数据
        
        Returns:
            list: 顶点坐标列表
        """
        return self.vertices
