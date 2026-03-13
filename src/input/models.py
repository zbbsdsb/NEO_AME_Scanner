class Sample:
    """
    Sample data structure containing coordinate and weight information
    """
    def __init__(self, x, y, z, weight=1.0):
        """
        Initialize Sample object
        
        Args:
            x: x coordinate
            y: y coordinate
            z: z coordinate
            weight: weight, default is 1.0
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.weight = float(weight)
    
    def to_dict(self):
        """
        Convert to dictionary format
        
        Returns:
            dict: Dictionary containing Sample fields
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'weight': self.weight
        }
    
    def to_tuple(self):
        """
        Convert to tuple format
        
        Returns:
            tuple: (x, y, z, weight)
        """
        return (self.x, self.y, self.z, self.weight)
    
    @classmethod
    def from_vertex(cls, vertex, weight=1.0):
        """
        Create Sample from vertex data
        
        Args:
            vertex: vertex coordinate tuple (x, y, z)
            weight: weight, default is 1.0
            
        Returns:
            Sample: Sample object
        """
        if len(vertex) < 3:
            raise ValueError("Vertex data must have at least 3 coordinate values")
        x, y, z = vertex[:3]
        return cls(x, y, z, weight)
    
    def __repr__(self):
        """
        String representation
        """
        return f"Sample(x={self.x}, y={self.y}, z={self.z}, weight={self.weight})"

class SampleCollection:
    """
    Sample collection management class
    """
    def __init__(self):
        """
        Initialize SampleCollection object
        """
        self.samples = []
    
    def add_sample(self, sample):
        """
        Add Sample to collection
        
        Args:
            sample: Sample object
        """
        if not isinstance(sample, Sample):
            raise TypeError("Only Sample type objects can be added")
        self.samples.append(sample)
    
    def add_samples_from_vertices(self, vertices, weight=1.0):
        """
        Add multiple Samples from vertex list
        
        Args:
            vertices: list of vertices, each element is (x, y, z) tuple
            weight: weight, default is 1.0
        """
        for vertex in vertices:
            sample = Sample.from_vertex(vertex, weight)
            self.add_sample(sample)
    
    def get_samples(self):
        """
        Get all Samples
        
        Returns:
            list: List of Sample objects
        """
        return self.samples
    
    def get_sample_count(self):
        """
        Get Sample count
        
        Returns:
            int: Number of Samples
        """
        return len(self.samples)
    
    def get_vertices(self):
        """
        Get all vertex coordinates
        
        Returns:
            list: List of vertex coordinates, each element is (x, y, z) tuple
        """
        return [(sample.x, sample.y, sample.z) for sample in self.samples]
    
    def to_list(self):
        """
        Convert to list of dictionaries
        
        Returns:
            list: List containing Sample dictionaries
        """
        return [sample.to_dict() for sample in self.samples]
    
    def __repr__(self):
        """
        String representation
        """
        return f"SampleCollection(sample_count={self.get_sample_count()})"

class SpatialEvidenceField:
    """
    Spatial Evidence Field data structure that unifies different input formats
    """
    def __init__(self):
        """
        Initialize SpatialEvidenceField object
        """
        self.samples = []
        self.bounds = None
        self.metadata = {}
    
    def add_sample(self, sample):
        """
        Add a sample to the field
        
        Args:
            sample: Sample object
        """
        if not isinstance(sample, Sample):
            raise TypeError("Only Sample type objects can be added")
        self.samples.append(sample)
        self._update_bounds(sample)
    
    def add_samples(self, samples):
        """
        Add multiple samples to the field
        
        Args:
            samples: List of Sample objects
        """
        for sample in samples:
            self.add_sample(sample)
    
    def add_samples_from_collection(self, collection):
        """
        Add samples from a SampleCollection
        
        Args:
            collection: SampleCollection object
        """
        if not isinstance(collection, SampleCollection):
            raise TypeError("Only SampleCollection type objects can be used")
        for sample in collection.get_samples():
            self.add_sample(sample)
    
    def get_samples(self):
        """
        Get all samples
        
        Returns:
            list: List of Sample objects
        """
        return self.samples
    
    def get_sample_count(self):
        """
        Get sample count
        
        Returns:
            int: Number of samples
        """
        return len(self.samples)
    
    def get_bounds(self):
        """
        Get spatial bounds of the field
        
        Returns:
            dict: Bounds with min and max coordinates
        """
        return self.bounds
    
    def get_metadata(self):
        """
        Get metadata
        
        Returns:
            dict: Metadata dictionary
        """
        return self.metadata
    
    def set_metadata(self, key, value):
        """
        Set metadata key-value pair
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def _update_bounds(self, sample):
        """
        Update spatial bounds when adding a new sample
        
        Args:
            sample: Sample object
        """
        if self.bounds is None:
            self.bounds = {
                'min_x': sample.x,
                'max_x': sample.x,
                'min_y': sample.y,
                'max_y': sample.y,
                'min_z': sample.z,
                'max_z': sample.z
            }
        else:
            self.bounds['min_x'] = min(self.bounds['min_x'], sample.x)
            self.bounds['max_x'] = max(self.bounds['max_x'], sample.x)
            self.bounds['min_y'] = min(self.bounds['min_y'], sample.y)
            self.bounds['max_y'] = max(self.bounds['max_y'], sample.y)
            self.bounds['min_z'] = min(self.bounds['min_z'], sample.z)
            self.bounds['max_z'] = max(self.bounds['max_z'], sample.z)
    
    def to_dict(self):
        """
        Convert to dictionary format
        
        Returns:
            dict: Dictionary representation of SpatialEvidenceField
        """
        return {
            'sample_count': self.get_sample_count(),
            'bounds': self.bounds,
            'metadata': self.metadata,
            'samples': [sample.to_dict() for sample in self.samples]
        }
    
    def __repr__(self):
        """
        String representation
        """
        return f"SpatialEvidenceField(sample_count={self.get_sample_count()}, bounds={self.bounds})"
