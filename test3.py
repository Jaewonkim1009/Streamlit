import pandas as pd
import numpy as np

# 데이터 불러오기
data = pd.read_csv('./data/Maternal Health Risk Data Set.csv')

# 임신 중 여성의 평균 키와 체중 데이터
average_height_cm = 163.4  # cm
average_weight_kg = 55.6  # kg

# 키와 체중 데이터를 평균 주변에서 임의로 생성
# 시드 고정
np.random.seed(42)
heights = np.random.normal(loc=average_height_cm, scale=6.2, size=len(data))  # cm
weights = np.random.normal(loc=average_weight_kg, scale=13.7, size=len(data))  # kg

# BMI 계산
bmi_values = weights / ((heights / 100) ** 2)

# 데이터프레임에 키, 체중, BMI, 임신 주수 추가
data['Height(cm)'] = heights
data['Weight(kg)'] = weights
data['BMI'] = bmi_values

# 임의로 생성된 임신주수 데이터
gestational_ages = np.random.randint(10, 41, size=len(data))  # 주

# 데이터프레임에 추가
data['GestationalAge'] = gestational_ages

# 업데이트된 데이터를 새로운 파일로 저장
data.to_csv('Updated_Maternal_Health_Risk_Data.csv', index=False)

# 데이터 확인
print(data.head())
