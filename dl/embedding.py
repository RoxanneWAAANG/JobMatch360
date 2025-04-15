import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
import time

class JinaEmbedV3:
    def __init__(self, model_path: str = 'jinaai/jina-embeddings-v3', onnx_path: str = 'jina-embeddings-v3/onnx/model.onnx'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = PretrainedConfig.from_pretrained(model_path)
        self.session = onnxruntime.InferenceSession(onnx_path)
    
    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray):
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def embed(self, text: str):
        start_time = time.time()
        # Prepare inputs for ONNX model
        input_text = self.tokenizer(text, return_tensors='np')
        task_type = 'text-matching'
        task_id = np.array(self.config.lora_adaptations.index(task_type), dtype=np.int64)
        inputs = {
            'input_ids': input_text['input_ids'],
            'attention_mask': input_text['attention_mask'],
            'task_id': task_id
        }
        # Run model
        outputs = self.session.run(None, inputs)[0]
        # Apply mean pooling and normalization to the model outputs
        embeddings = self._mean_pooling(outputs, input_text["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        end_time = time.time()
        return embeddings, end_time - start_time

if __name__ == "__main__":
    sample_text = "Rock N Roll Sushi is hiring a Restaurant Manager!\nAs our Restaurant Manager, you’ll never be bored. You’ll be responsible for making sure our restaurant runs smoothly.\nWe Offer\nCompetitive compensation\nInsurance benefits\nBonus opportunities\nA great work atmosphere\nDuties/Responsibilities\nEnsuring that our restaurant is fully and appropriately staffed at all times\nMaintaining operational excellence so our restaurant is running efficiently and effectively\nEnsuring that all laws, regulations, and guidelines are being followed\nCreating a restaurant atmosphere that both patrons and employees enjoy\nVarious other tasks as needed\nRequirements\nPrevious experience as a restaurant manager\nExtensive food and beverage knowledge, and the ability to remember and recall ingredients and dishes to inform customers and wait staff\nGreat leadership skills\nFamiliarity with restaurant management software\nDemonstrated ability to coordinate a staff\nShow more\nShow less"
    embedder = JinaEmbedV3()
    embeddings, time_taken = embedder.embed(sample_text)
    print(type(embeddings), embeddings.shape)
    print(embeddings)
    print(f"Time taken: {time_taken} seconds")
