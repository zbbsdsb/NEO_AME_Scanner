class PLYLoader:
    def __init__(self):
        self.vertices = []
        self.sample_count = 0
        self.file_path = None
    
    def load(self, file_path):
        """
        Load PLY file
        
        Args:
            file_path: PLY file path
            
        Returns:
            list: List of vertex coordinates, each element is (x, y, z) tuple
            
        Raises:
            FileNotFoundError: File not found
            ValueError: File format error
        """
        self.file_path = file_path
        self.vertices = []
        
        try:
            with open(file_path, 'r') as f:
                # Parse header
                header_info = self._parse_header(f)
                vertex_count = header_info['vertex_count']
                
                # Read vertex data
                self.vertices = self._read_vertices(f, vertex_count)
                
                # Validate vertex count
                if len(self.vertices) != vertex_count:
                    raise ValueError(f"Vertex count mismatch: expected {vertex_count}, actual {len(self.vertices)}")
                
                self.sample_count = len(self.vertices)
                return self.vertices
                
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to parse PLY file: {str(e)}")
    
    def _parse_header(self, file):
        """
        Parse PLY file header
        
        Args:
            file: File object
            
        Returns:
            dict: Header information, including vertex_count etc.
        """
        header_info = {}
        
        # Read first line, validate file type
        first_line = file.readline().strip()
        if first_line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        # Read format information
        format_line = file.readline().strip()
        if not format_line.startswith('format ascii'):
            raise ValueError("Only ASCII format PLY files are supported")
        
        # Read other header information
        while True:
            line = file.readline().strip()
            if line == 'end_header':
                break
            
            if line.startswith('element vertex'):
                # Extract vertex count
                parts = line.split()
                if len(parts) >= 3:
                    header_info['vertex_count'] = int(parts[2])
            
            # Ignore other header information
        
        if 'vertex_count' not in header_info:
            raise ValueError("PLY file header missing vertex information")
        
        return header_info
    
    def _read_vertices(self, file, vertex_count):
        """
        Read vertex data
        
        Args:
            file: File object
            vertex_count: Number of vertices
            
        Returns:
            list: List of vertex coordinates
        """
        vertices = []
        
        for i in range(vertex_count):
            line = file.readline().strip()
            if not line:
                raise ValueError(f"Insufficient vertex data, expected {vertex_count} vertices")
            
            # Split line data
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Vertex data format error, at least 3 coordinate values are required")
            
            try:
                # Only read first three values as x, y, z coordinates
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                vertices.append((x, y, z))
            except ValueError:
                raise ValueError(f"Vertex coordinate values are not valid numbers")
        
        return vertices
    
    def get_sample_count(self):
        """
        Get sample count
        
        Returns:
            int: Number of samples
        """
        return self.sample_count
    
    def get_vertices(self):
        """
        Get loaded vertex data
        
        Returns:
            list: List of vertex coordinates
        """
        return self.vertices
