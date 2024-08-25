import json
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import os
from fuzzywuzzy import process
from flask import Flask, request, jsonify

app = Flask(__name__)

# 讀取 JSON 檔案中的數據
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 初始化微調的問答模型
def init_pert_model():
    model_name = "uer/roberta-base-chinese-extractive-qa"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

# 使用模糊匹配來取得相關文檔
def search_documents(query, data):
    documents = [doc['content'] for doc in data]
    relevant_docs = [content for content, score in process.extract(query, documents, limit=5) if score > 50]
    return relevant_docs

# 生成答案
def generate_answer(question, data, tokenizer, model):
    docs = search_documents(question, data)
    if not docs:
        return "找不到相關文檔。"
    
    context = " ".join(docs)
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    outputs = model(**inputs)
    
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    if answer_start >= answer_end:
        return "無法生成合理答案。"
    
    answer_ids = inputs["input_ids"][0][answer_start:answer_end].tolist()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer

# 初始化模型和數據
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'data.json')
data = read_data(data_file_path)
tokenizer, model = init_pert_model()

# 定義 API 路由
@app.route('/qa', methods=['POST'])
def qa():
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({"error": "問題未提供"}), 400
        
        answer = generate_answer(question, data, tokenizer, model)
        response = {"question": question, "answer": answer}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 啟動 Flask 服務
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=66, debug=True)
