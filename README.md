Download csv files from [kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/) to `data/`.

Run commands below to download jina-v3 embedding model weight.
```bash
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx -O jina-embeddings-v3/onnx/model.onnx
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data -O jina-embeddings-v3/onnx/model.onnx_data
wget https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model_fp16.onnx -O jina-embeddings-v3/onnx/model_fp16.onnx
```

Add your resume to root directory then run command below to get recommendation.
```bash
python recommand.py
```