import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine

# 데이터 개수
num_samples = 1000

# 랜덤 시드 고정
np.random.seed(42)

# --------------------
# 산모 건강 관련 데이터 생성
# --------------------
# 임신 주수 먼저 생성 (후속 계산에 필요)
pregnancy_weeks = np.random.randint(5, 41, num_samples)

# 초산/경산 여부 생성 (태아 체중에 영향)
is_first_pregnancy = np.random.choice([True, False], num_samples, p=[0.45, 0.55])

# 태아 성별 (34주 이후 남아가 여아보다 100g 정도 무거움)
fetal_gender = np.random.choice(['male', 'female'], num_samples, p=[0.51, 0.49])

# 태아 체중 계산 함수 (한국 신생아 성장 패턴 반영)
def calculate_fetal_weight(week, gender, first_pregnancy):
    # 29-38주까지 직선에 가까운 성장 패턴 반영
    if week < 29:
        base_weight = 500 + (week * 50)  # 초기 성장
    elif 29 <= week <= 38:
        # 29주부터 38주까지 직선적인 성장
        base_weight = 1400 + ((week - 29) * 200)  # 29주에 약 1400g, 매주 약 200g 증가
    else:
        # 38주 이후 성장률 감소
        base_weight = 3200 + ((week - 38) * 150)
    
    # 34주 이후 성별에 따른 차이 (남아가 약 100g 더 무거움)
    if week >= 34 and gender == 'male':
        base_weight += 100
    
    # 초산부의 경우 50-100g 감소
    if first_pregnancy:
        base_weight -= np.random.randint(50, 101)
    
    # 개인차 반영 (±10% 변동)
    variation = np.random.uniform(-0.1, 0.1)
    weight = base_weight * (1 + variation)
    
    return max(500, int(weight))  # 최소 500g 보장

# 태아 체중 계산
fetal_weights = [calculate_fetal_weight(week, gender, first) 
                 for week, gender, first in zip(pregnancy_weeks, fetal_gender, is_first_pregnancy)]

# 출생 체중 계산 (예상 40주 체중에 가까움)
birth_weights = []
for i in range(num_samples):
    if pregnancy_weeks[i] >= 37:  # 만삭아
        birth_weights.append(fetal_weights[i] + np.random.randint(100, 301))
    else:  # 조산아
        # 40주까지 예상 성장을 고려한 체중
        expected_weight = calculate_fetal_weight(40, fetal_gender[i], is_first_pregnancy[i])
        birth_weights.append(int(fetal_weights[i] * 0.9))  # 현재 태아 체중보다 약간 작게 설정

data = {
    # 기본 정보
    'mother_id': [f'M{str(i).zfill(4)}' for i in range(1, num_samples + 1)],
    'age': np.random.randint(20, 45, num_samples),
    'bmi': np.round(np.random.uniform(18.5, 35, num_samples), 1),
    'pregnancy_week': pregnancy_weeks,
    'is_first_pregnancy': is_first_pregnancy,
    'fetal_gender': fetal_gender,
    
    # 건강 지표
    'blood_pressure_sbp': np.random.randint(90, 160, num_samples),
    'blood_pressure_dbp': np.random.randint(60, 100, num_samples),
    'blood_sugar': np.random.randint(70, 200, num_samples),
    'weight_gain': np.round(np.random.uniform(-1, 20, num_samples), 1),
    'caloric_intake': np.random.randint(1500, 3000, num_samples),
    'exercise_per_week': np.random.randint(0, 7, num_samples),
    'sleep_hours': np.round(np.random.uniform(4, 10, num_samples), 1),
    'stress_level': np.random.randint(1, 10, num_samples),
    'smoking': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
    'alcohol': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
    
    # 태아 건강 지표 (한국 신생아 특성 반영)
    'fetal_weight': fetal_weights,
    'fetal_heart_rate': np.random.randint(110, 170, num_samples),
    'fetal_movements': np.random.randint(1, 10, num_samples),
    'amniotic_fluid_index': np.round(np.random.uniform(5, 25, num_samples), 1),
    'preterm_risk': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
    'birth_weight': birth_weights,
    
    # 유전적 요인
    'family_history_hypertension': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    'family_history_diabetes': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
    'family_history_heart_disease': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
    'family_history_preterm': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
    'family_history_twins': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
    'genetic_disorder': np.random.choice([0, 1], num_samples, p=[0.98, 0.02])
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# --------------------
# MySQL 저장 설정
# --------------------
DB_HOST = 'localhost'
DB_NAME = 'maternal_db'
DB_USER = 'root'
DB_PASS = '1234'
DB_TABLE = 'maternal_health'
DB_PORT = '3306'

# SQLAlchemy 엔진 생성
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# 데이터 MySQL에 저장
df.to_sql(DB_TABLE, con=engine, if_exists='replace', index=False)