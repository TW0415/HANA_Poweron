#_*_ coding:utf-8
import threading
import json

from utils.DataBase import Database
from model.QA.QAModel import QAmodel
from model.classify.Classifymodel import ClassifyModel
from transformers import AutoTokenizer

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", 
                                          do_lower_case=False,
                                          use_fast=False)


# 조항 분류 모델
classify = ClassifyModel('/workspace/Hana_real/model/model_classification.pt',tokenizer)

print("약관 내용이 궁금한 상품을 입력해주세요")

input_fund = "하나UBS행복한TDF2045"
print(input_fund)

print("어떤 내용이 궁금하신가요?")

input_query = "집합투자업자의 업무란"

print(input_query)

json_data = {
    		'Query': input_query,
    		'fund': input_fund,
			}


input_data = json.loads(json.dumps(json_data, ensure_ascii=False))

query = input_data['Query']

fund = input_data['fund']


class_predict = classify.predict_class(query)


# DB에서 해당되는 조항 탐색
DB = Database("mongodb 주소")

DB_values = json.loads(DB.Search(class_predict, fund))

# QA 모델

QA = QAmodel("/workspace/Hana_real/model/model_QA.pt",
             DB_values,
             tokenizer)

# 답변 검색
message = QA.QA_search(query)

print(message)