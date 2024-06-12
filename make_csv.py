"""
실수로 생성된 solution을 csv 파일로 저장하는 코드를 안 넣어서
별도로 리스트를 넣어서 csv 파일을 생성
"""
import pandas as pd
import csv

# 리스트 예시
data = [1, 2, 3]

# CSV 파일 경로
csv_file = 'test.csv'

# list를 DataFrame으로 변환
df = pd.DataFrame(data)

# Dataframe을 CSV 파일로 저장
df.to_csv(csv_file, index=False, header=False)
