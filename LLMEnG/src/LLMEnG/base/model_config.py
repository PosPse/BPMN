from enum import Enum
class LLMConfig():
    def __init__(self, model_name, model_path, hidden_size):
        self.model_name = model_name
        self.model_path = model_path
        self.hidden_size = hidden_size

class ModelType(Enum):
    Bert_Base = "bert-base-uncased"
    Bert_Largr = "bert-large-uncased" 

class ModelConfig():
    bert_base = LLMConfig("bert-base-uncased", "/home/btr/bpmn/model/safetensors/bert-base-uncased", 768)
    bert_large = LLMConfig("bert-large-uncased", "/home/btr/bpmn/model/safetensors/bert-large-uncased", 1024)

    models = {
        "bert-base-uncased": bert_base,
        "bert-large-uncased": bert_large
    }

    current_model = "bert-base-uncased"

    @staticmethod
    def set_current_model(model_name):
        if model_name in ModelConfig.models:
            ModelConfig.current_model = model_name
        else:
            raise ValueError(f"Model {model_name} not found")
        
    @staticmethod
    def get_current_model():
        return ModelConfig.models[ModelConfig.current_model]

