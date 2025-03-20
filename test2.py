import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine

# MySQL 연결 정보
DB_HOST = 'localhost'
DB_NAME = 'maternal_db'
DB_USER = 'root'
DB_PASS = '1234'
DB_TABLE = 'maternal_health'
DB_PORT = '3306'

# 페이지 설정
st.set_page_config(page_title='산모 건강 분석 대시보드', page_icon=':pregnant_woman:', layout='wide')

# MySQL 연결 함수 생성
def create_database_connection():
    try:
        # 데이터베이스 연결
        connection = mysql.connector.connect(
            host = DB_HOST,
            database = DB_NAME,
            user = DB_USER,
            password = DB_PASS
        )
        if connection.is_connected():
            print('MySQL 데이터베이스에 성공적으로 연결되었습니다.')
        return connection
    except Error as e:
        print(f'데이터베이스 연결 중 오류 발생: {e}')
        return None

# SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# 데이터 가져오기 (캐싱 적용)
@st.cache_data
def get_data():
    connection = create_database_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {DB_TABLE}"
        df = pd.read_sql(query, connection)
        return df
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()

# 필터링된 데이터 캐싱
@st.cache_data
def filter_data(df, age_range, bmi_range, pregnancy_week_range):
    filtered_df = df.copy()
    
    # 나이 필터링
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    # BMI 필터링
    filtered_df = filtered_df[(filtered_df['bmi'] >= bmi_range[0]) & (filtered_df['bmi'] <= bmi_range[1])]
    
    # 임신 주차 필터링
    filtered_df = filtered_df[(filtered_df['pregnancy_week'] >= pregnancy_week_range[0]) & 
                              (filtered_df['pregnancy_week'] <= pregnancy_week_range[1])]
    
    return filtered_df

# 메인 함수
def main():
    # 제목
    st.title('산모 건강 분석 대시보드')
    st.markdown('산모의 건강 지표와 유전적 요인이 태아에게 미치는 영향과 조산 위험 분석')
    
    # 데이터 불러오기
    df = get_data()
    
    if df.empty:
        st.error("데이터베이스 연결 오류 또는 데이터를 불러올 수 없습니다.")
        return
    
    # 사이드바 필터 생성
    st.sidebar.title('데이터 필터링')
    
    # 나이 범위 선택
    age_min = int(df['age'].min())
    age_max = int(df['age'].max())
    age_range = st.sidebar.slider(
        '산모 나이 범위:',
        age_min, age_max, (age_min, age_max)
    )
    
    # BMI 범위 선택
    bmi_min = float(df['bmi'].min())
    bmi_max = float(df['bmi'].max())
    bmi_range = st.sidebar.slider(
        'BMI 범위:',
        bmi_min, bmi_max, (bmi_min, bmi_max)
    )
    
    # 임신 주차 범위 선택
    week_min = int(df['pregnancy_week'].min())
    week_max = int(df['pregnancy_week'].max())
    pregnancy_week_range = st.sidebar.slider(
        '임신 주차 범위:',
        week_min, week_max, (week_min, week_max)
    )
    
    # 데이터 필터링 적용
    filtered_df = filter_data(df, age_range, bmi_range, pregnancy_week_range)
    
    # 필터링된 데이터 수 표시
    st.sidebar.markdown(f"**필터링된 데이터:** {filtered_df.shape[0]} 건")
    
    # 데이터 개요 표시
    with st.expander("필터링된 데이터 미리보기"):
        st.dataframe(filtered_df.head(10))
    
    # 주요 지표 표시 (3개 컬럼)
    col1, col2, col3 = st.columns(3)
    
    # 평균 나이
    avg_age = round(filtered_df['age'].mean(), 1)
    col1.metric(label='평균 나이', value=f'{avg_age}세')
    
    # 평균 BMI
    avg_bmi = round(filtered_df['bmi'].mean(), 1)
    col2.metric(label='평균 BMI', value=f'{avg_bmi}')
    
    # 조산 위험 비율
    preterm_risk_pct = round(filtered_df['preterm_risk'].mean() * 100, 1)
    col3.metric(label='조산 위험 비율', value=f'{preterm_risk_pct}%')
    
    st.divider()
    
    # ---------------------------------------
    # 시각화 섹션
    # ---------------------------------------
    
    # 1. 나이별 조산 위험 분포 (막대 그래프)
    st.subheader('1. 나이별 조산 위험 분포')
    
    age_risk_df = filtered_df.groupby('age')['preterm_risk'].mean().reset_index()
    age_risk_df['preterm_risk_pct'] = age_risk_df['preterm_risk'] * 100
    
    fig_age_risk = px.bar(
        age_risk_df, 
        x='age', 
        y='preterm_risk_pct',
        labels={'age': '산모 나이', 'preterm_risk_pct': '조산 위험 확률 (%)'},
        title='나이별 조산 위험 확률',
        color='preterm_risk_pct',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig_age_risk, use_container_width=True)
    
    # 2행 2열 레이아웃
    col1, col2 = st.columns(2)
    
    # 2. BMI와 혈압의 관계 (산점도)
    col1.subheader('2. BMI와 혈압의 관계')
    fig_bmi_bp = px.scatter(
        filtered_df, 
        x='bmi', 
        y='blood_pressure_sbp',
        color='preterm_risk',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'bmi': 'BMI', 'blood_pressure_sbp': '수축기 혈압', 'preterm_risk': '조산 위험'},
        title='BMI와 수축기 혈압의 관계',
        hover_data=['age', 'blood_pressure_dbp']
    )
    col1.plotly_chart(fig_bmi_bp, use_container_width=True)
    
    # 3. 태아 체중과 심박수 관계 (산점도)
    col2.subheader('3. 태아 체중과 심박수 관계')
    fig_fetal = px.scatter(
        filtered_df, 
        x='fetal_weight', 
        y='fetal_heart_rate',
        color='preterm_risk',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'fetal_weight': '태아 체중 (g)', 'fetal_heart_rate': '태아 심박수', 'preterm_risk': '조산 위험'},
        title='태아 체중과 심박수의 관계',
        hover_data=['pregnancy_week']
    )
    col2.plotly_chart(fig_fetal, use_container_width=True)
    
    st.divider()
    
    # 4. 유전적 요인별 조산 위험 (막대 그래프)
    st.subheader('4. 유전적 요인별 조산 위험')
    
    # 유전적 요인 열 이름 목록
    genetic_factors = [
        'family_history_hypertension', 
        'family_history_diabetes', 
        'family_history_heart_disease',
        'family_history_preterm', 
        'family_history_twins', 
        'genetic_disorder'
    ]
    
    # 각 유전적 요인별 조산 위험 계산
    genetic_risk_data = []
    
    for factor in genetic_factors:
        # 요인이 있는 경우 (1)의 조산 위험
        risk_with_factor = filtered_df[filtered_df[factor] == 1]['preterm_risk'].mean() * 100
        
        # 요인이 없는 경우 (0)의 조산 위험
        risk_without_factor = filtered_df[filtered_df[factor] == 0]['preterm_risk'].mean() * 100
        
        # 가독성을 위한 요인 이름 변경
        factor_name = factor.replace('family_history_', '').replace('_', ' ').title()
        
        genetic_risk_data.append({
            'factor': factor_name,
            'risk_with_factor': risk_with_factor,
            'risk_without_factor': risk_without_factor
        })
    
    genetic_risk_df = pd.DataFrame(genetic_risk_data)
    
    fig_genetic = go.Figure()
    
    fig_genetic.add_trace(go.Bar(
        x=genetic_risk_df['factor'],
        y=genetic_risk_df['risk_with_factor'],
        name='요인 있음',
        marker_color='red'
    ))
    
    fig_genetic.add_trace(go.Bar(
        x=genetic_risk_df['factor'],
        y=genetic_risk_df['risk_without_factor'],
        name='요인 없음',
        marker_color='blue'
    ))
    
    fig_genetic.update_layout(
        title='유전적 요인 유무에 따른 조산 위험 확률',
        xaxis_title='유전적 요인',
        yaxis_title='조산 위험 확률 (%)',
        barmode='group'
    )
    
    st.plotly_chart(fig_genetic, use_container_width=True)
    
    # 5. 생활 습관과 조산 위험 (2열 레이아웃)
    col1, col2 = st.columns(2)
    
    # 5-1. 수면 시간과 조산 위험
    col1.subheader('5. 수면 시간과 조산 위험')
    sleep_risk_df = filtered_df.groupby('sleep_hours')['preterm_risk'].mean().reset_index()
    sleep_risk_df['preterm_risk_pct'] = sleep_risk_df['preterm_risk'] * 100
    
    fig_sleep = px.line(
        sleep_risk_df,
        x='sleep_hours',
        y='preterm_risk_pct',
        labels={'sleep_hours': '수면 시간 (시간)', 'preterm_risk_pct': '조산 위험 확률 (%)'},
        title='수면 시간에 따른 조산 위험',
        markers=True
    )
    col1.plotly_chart(fig_sleep, use_container_width=True)
    
    # 5-2. 스트레스 레벨과 조산 위험
    col2.subheader('6. 스트레스 레벨과 조산 위험')
    stress_risk_df = filtered_df.groupby('stress_level')['preterm_risk'].mean().reset_index()
    stress_risk_df['preterm_risk_pct'] = stress_risk_df['preterm_risk'] * 100
    
    fig_stress = px.line(
        stress_risk_df,
        x='stress_level',
        y='preterm_risk_pct',
        labels={'stress_level': '스트레스 레벨', 'preterm_risk_pct': '조산 위험 확률 (%)'},
        title='스트레스 레벨에 따른 조산 위험',
        markers=True
    )
    col2.plotly_chart(fig_stress, use_container_width=True)
    
    # 7. 운동 빈도와 칼로리 섭취량에 따른 태아 체중 관계 (개선된 버전)
    st.subheader('7. 운동 빈도와 칼로리 섭취량에 따른 태아 체중')
    
    # 운동 빈도를 범주화 (0-1, 2-3, 4-5, 6-7)
    filtered_df['exercise_category'] = pd.cut(
        filtered_df['exercise_per_week'],
        bins=[0, 1, 3, 5, 7],
        labels=['0-1회/주', '2-3회/주', '4-5회/주', '6-7회/주']
    )
    
    # 칼로리 섭취량을 범주화 (500단위)
    filtered_df['caloric_category'] = pd.cut(
        filtered_df['caloric_intake'],
        bins=[1500, 2000, 2500, 3000],
        labels=['1500-2000', '2000-2500', '2500-3000']
    )
    
    # 범주화된 값으로 그룹화
    exercise_calories_df = filtered_df.groupby(['exercise_category', 'caloric_category']).agg({
        'fetal_weight': 'mean'
    }).reset_index()
    
    # 히트맵 생성
    exercise_calories_pivot = exercise_calories_df.pivot_table(
        values='fetal_weight',
        index='exercise_category',
        columns='caloric_category'
    ).round(0)
    
    # 히트맵 그래프 생성
    fig_heatmap = px.imshow(
        exercise_calories_pivot,
        text_auto=True,  # 셀 내에 값 표시
        labels=dict(x="칼로리 섭취량", y="운동 빈도", color="태아 체중 (g)"),
        title="운동 빈도와 칼로리 섭취량에 따른 태아 체중",
        color_continuous_scale="Viridis",
        aspect="auto"  # 자동 비율 조정
    )
    
    # 히트맵 레이아웃 조정
    fig_heatmap.update_layout(
        xaxis={'title': {'text': '칼로리 섭취량 (kcal)'}},
        yaxis={'title': {'text': '운동 빈도'}},
        coloraxis_colorbar={'title': '태아 체중 (g)'},
        height=500,  # 높이 조정
        font=dict(size=14)  # 글꼴 크기 조정
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 8. 임신 주차별 태아 체중 변화 (선 그래프)
    st.subheader('8. 임신 주차별 태아 체중 변화')
    
    # 임신 주차별 태아 체중 평균
    week_weight_df = filtered_df.groupby('pregnancy_week').agg({
        'fetal_weight': 'mean',
        'preterm_risk': 'mean'
    }).reset_index()
    
    # 그래프 생성
    fig_week_weight = px.line(
        week_weight_df,
        x='pregnancy_week',
        y='fetal_weight',
        labels={'pregnancy_week': '임신 주차', 'fetal_weight': '태아 체중 (g)'},
        title='임신 주차별 태아 체중 변화',
        markers=True
    )
    
    # 조산 위험이 높은 주차 표시
    high_risk_weeks = week_weight_df[week_weight_df['preterm_risk'] > 0.2]['pregnancy_week'].tolist()
    
    for week in high_risk_weeks:
        fig_week_weight.add_vline(
            x=week, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"위험주차: {week}주",
            annotation_position="top right"
        )
    
    st.plotly_chart(fig_week_weight, use_container_width=True)
    
    # 푸터
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        footer:after{
            content: '산모 건강 분석 대시보드';
            visibility: visible;
            position: relative;
            text-align: center;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()