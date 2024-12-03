import os

import torch

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.utils.hub import cached_file

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class LLMEmbedding:
    def __init__(self, model_name: str):
        """
        初始化 LLaMA 或 BERT 模型
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 根据模型名称加载不同的模型
        if "llama" in model_name.lower():
            self.model_type = "llama"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, output_hidden_states=True
            )
        elif "bert" in model_name.lower():
            self.model_type = "bert"
            self.model = AutoModel.from_pretrained(
                model_name, output_hidden_states=True
            )
        else:
            raise ValueError(
                "Unsupported model type. Only LLaMA and BERT are supported."
            )

        print(f"Model type detected: {self.model_type}")

    def compose_prompt(self, text: str) -> str:
        """
        构造 LLaMA 的生成任务 prompt
        """
        if self.model_type == "llama":
            return f"""下方是一个问题，请针对这个问题给出你的回答。请只回答这个问题，不要生成其他内容。\n\n### 问题：\n{text}\n\n### 回答："""
        elif self.model_type == "bert":
            return text  # BERT 不需要特殊 prompt
        else:
            raise ValueError("Unsupported model type.")

    def get_embedding(self, text: str):
        """
        获取文本的 embedding 和生成的输出。

        Args:
            text (str): 输入文本。
            pool_method (str): 提取 embedding 的方式：
                - "cls": 仅适用于 BERT，提取 [CLS] token 的隐藏状态。
                - "mean": 对所有 token 的隐藏状态进行平均池化。
                - "last": 仅适用于 LLaMA，提取最后一个 token 的隐藏状态。

        Returns:
            embedding (np.ndarray): 提取的文本 embedding。
            model_output (str): 模型生成的文本（仅适用于 LLaMA）。
        """
        # Tokenize 输入文本
        inputs = self.tokenizer(text, return_tensors="pt")

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取 embedding
        last_hidden_state = outputs.hidden_states[-1]  # 获取最后一层隐藏状态
        if self.model_type == "bert":
            # if pool_method == "cls":
            embedding = (
                last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            )  # [CLS] token
            # elif pool_method == "mean":
            #     embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 平均池化
            # else:
            #     raise ValueError("Invalid pool_method for BERT. Use 'cls' or 'mean'.")
            model_output = ""  # BERT 不支持生成任务
        elif self.model_type == "llama":
            # if pool_method == "last":
            embedding = (
                last_hidden_state[:, -1, :].squeeze().cpu().numpy()
            )  # 最后一个 token
            # elif pool_method == "mean":
            #     embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 平均池化
            # else:
            #     raise ValueError("Invalid pool_method for LLaMA. Use 'last' or 'mean'.")
            # 使用 generate 方法生成文本
            output_ids = self.model.generate(inputs.input_ids, max_new_tokens=50)
            model_output = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
        else:
            raise ValueError("Unsupported model type.")

        return embedding, model_output

    def get_embeddings(self, variable_text_map: dict, target_folder: str):
        """
        批量获取 embedding 和生成文本，并保存结果。

        Args:
            variable_text_map (dict): 变量名与问题的映射。
            target_folder (str): 保存结果的目标文件夹。
            pool_method (str): 提取 embedding 的方式。
        """
        model_results = []
        embedding_dict = {}

        for variable, question in variable_text_map.items():
            # 构造 prompt 并获取结果
            prompt = self.compose_prompt(question)
            embedding, model_output = self.get_embedding(prompt)

            # 保存结果
            model_results.append(
                {
                    "variable": variable,
                    "question": question,
                    "model_output": model_output,
                }
            )
            embedding_dict[variable] = embedding

        # 如果目标文件夹不存在，则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 保存模型输出结果到 CSV
        df = pd.DataFrame(model_results)
        results_path = os.path.join(target_folder, "model_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Model results saved to {results_path}")

        # 保存 embedding 到 JSON 文件
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
    # model_name = "hfl/llama-3-chinese-8b"
    model_name = "hfl/llama-3-chinese-8b-instruct-v3"
    # model_name = "hfl/chinese-roberta-wwm-ext"
    # model_name = "bert-base-chinese"

    embedding_generator = LLMEmbedding(model_name)
    embedding_generator.get_embeddings(variable_to_question, model_name)
