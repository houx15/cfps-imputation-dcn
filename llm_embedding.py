import os

import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from transformers.utils.hub import cached_file

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class LLMEmbedding(object):
    def __init__(self, model_name: str = "meta-llama/Llama-2-13b-hf"):
        self.model_name = model_name
        model_cache_path = cached_file(model_name, "config.json")
        print(f"Model cache path: {model_cache_path}")

        # 获取缓存根路径
        cache_root = os.path.dirname(os.path.dirname(model_cache_path))
        print(f"Cache root directory: {cache_root}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def compose_prompt(self, text):
        return (
            f"""下方是一个问题，请回答这个问题。\n\n### 问题：\n{text}\n\n### 回答："""
        )

    def get_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        if "bert" in self.model_name:
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        else:
            embedding = outputs.last_hidden_state[:, -1, :].squeeze().cpu().numpy()

        model_output = self.tokenizer.decode(
            torch.argmax(outputs.last_hidden_state, dim=-1).squeeze(),
            skip_special_tokens=True,
        )

        return embedding, model_output

    def get_embeddings(self, variable_text_map: dict, target_folder: str):
        model_results = []
        embedding_dict = {}
        for variable, question in variable_text_map.items():
            embedding, model_output = self.get_embedding(self.compose_prompt(question))
            model_results.append(
                {
                    "variable": variable,
                    "question": question,
                    "model_output": model_output,
                }
            )
            embedding_dict[variable] = embedding

        # 如果folder不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        df = pd.DataFrame(model_results)
        df.to_csv(os.path.join(target_folder, "model_results.csv"), index=False)
        print(
            f"Model results saved to {os.path.join(target_folder, 'model_results.csv')}"
        )

        np.save(os.path.join(target_folder, "embeddings.npy"), embedding_dict)
        print(f"Embeddings saved to {os.path.join(target_folder, 'embeddings.npy')}")


# 示例用法
if __name__ == "__main__":
    merge_info_data = pd.read_csv("data/cfps-merge.csv")
    merge_info_data = merge_info_data[merge_info_data["keep"] == 1]
    variable_to_question = {}
    for index, row in merge_info_data.iterrows():
        variable_to_question[row["variable"]] = row["question"]

    # 模型名称
    model_name = "hfl/llama-3-chinese-8b"
    model_name = "hfl/chinese-roberta-wwm-ext"
    model_name = "bert-base-chinese"

    embedding_generator = LLMEmbedding(model_name)
    embedding_generator.get_embeddings(variable_to_question, model_name)
