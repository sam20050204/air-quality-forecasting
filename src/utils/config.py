import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / 'configs'
        
        # Load configurations
        self.model_config = self._load_yaml('model_config.yaml')
        self.data_config = self._load_yaml('data_config.yaml')
        self.app_config = self._load_yaml('app_config.yaml')
    
    def _load_yaml(self, filename):
        with open(self.config_dir / filename, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def pollutants(self):
        return self.data_config['pollutants']
    
    @property
    def aqi_thresholds(self):
        return self.app_config['aqi_thresholds']

# Singleton instance
config = Config()