import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine

# MySQL 연결 정보
DB_HOST = 'localhost'
DB_NAME = 'tabledb'
DB_USER = 'root'
DB_PASS = '1234'
DB_TABLE = 'cars' # 테이블 이름
DB_PORT = '3306'
# 페이지 설정
st.set_page_config(page_title='자동차 재고 현황', page_icon=':racing_car:', layout='wide')

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
    # from mysql.connector import Error
    except Error as e:
        print(f'데이터베이스 연결 중 오류 발생: {e}')
        return None
    

# SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# # 데이터 가져오기 함수
# @st.cache_data
# def get_data():
#     connection = create_database_connection()
#     query = 'select * from tabledb.cars;'
#     df = pd.read_sql(query, connection) # SQLAlchemy 사용
#     return df
# st.title('자동차 재고 현황')
# df = get_data()
# st.dataframe(df.head(20))

# 데이터 가져오기 (캐싱 적용)
@st.cache_data
def get_data():
    connection = create_database_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {DB_TABLE}"
        df = pd.read_sql(query, connection)
        # 열 이름 조정
        df = df.rename(columns={
            "foreign_local_used": "Foreign_Local_Used",
            "seat_make": "seat-make",
            "make_year": "make-year",
            "automation": "Automation"  # 추가: 소문자 -> 대문자
        })
        return df
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()

# 필터링된 데이터 캐싱
@st.cache_data
def filter_data(df, manufacturer, automation, use_category):
    return df.query(
        "manufacturer == @manufacturer & Automation == @automation & Foreign_Local_Used == @use_category"
    )

df = get_data()

# 사이드바 만들기
st.sidebar.title('항목을 선택해주세요.')

# 제조사 선택
manufacturer = st.sidebar.multiselect(
    '제조사를 선택해주세요',
    options=df['manufacturer'].unique(),
    default=df['manufacturer'].loc[0]
)
# 변속기 선택
automation = st.sidebar.radio('변속기를 선택해주세요.',
                              options=df['Automation'].unique())

# 카테고리 선택
category = st.sidebar.radio('카테고리를 선택해주세요.',
                        options=df['Foreign_Local_Used'].unique())

# 필터링하기
df_select = filter_data(df, manufacturer, automation, category)

# 메인 화면
st.title('자동차 재고 현황')
st.dataframe(df_select)

# 평균가격, 수량, 최초 생산년도 순으로 칼럼 생성
col1, col2, col3 = st.columns(3)

# 평균가격
avg_price = int((df_select['price'] / 1000).mean())
col1.metric(label='평균 가격', value=f'US ${avg_price}')
# 수량
car_count = df_select.shape[0]
col2.metric(label='수량', value=f'{car_count}' + "대")
# 최초 생산년도
make_year = df_select['make-year'].min()
col3.metric(label='최초 생산 년도', value=f'{make_year}' + '년')

# 선 추가
st.divider()

# 색상 별 가격 막대 차트
fig_color_price = px.bar(df_select, x='color', y='price', title='색상별 평균 가격',
                         labels={'price': '가격', 'color': '색상'}, color='color', barmode='group')
st.plotly_chart(fig_color_price)
# 제조사 별 가격 막대 차트
fig_manufacturer_price = px.bar(df_select, x='manufacturer', y='price', title='제조사별 평균 가격',
                                labels={'price': '가격', 'manufacturer': '제조사'}, color='manufacturer', barmode='group')
st.plotly_chart(fig_manufacturer_price)
# 차트 화면 표시 (2열)
col1, col2 = st.columns(2)
# 차량 수량 히스토그램
col1.subheader('제조사별 차량 수량')
fig_car_count = px.histogram(df_select, x='manufacturer', title='제조사별 차량 수량', labels={'manufacturer': '제조사'})
col1.plotly_chart(fig_car_count)

# 가격 분포 박스 플롯
col2.subheader('가격 분포')
fig_price_hist = px.histogram(df_select, x='price', title='가격 분포', labels={'price': '가격'}, nbins=30, color='price')
col2.plotly_chart(fig_price_hist)

st.divider()

# 시트 유형별 파이 차트
fig_seat_pie = px.pie(df_select, names='seat-make', values='price', title='시트 유형별 가격 비율',
                       labels={'seat-make': '시트 유형', 'price': '가격'})
st.plotly_chart(fig_seat_pie)
# 가격 지표

# 3열 레이아웃
col1, col2, col3 = st.columns(3)

# 평균 가격
col1.metric(label='최고 가격', value=f'US ${df_select["price"].max()}')
# 최저 가격
col2.metric(label='최저 가격', value=f'US ${df_select["price"].min()}')
# 중간 가격
col3.metric(label='중앙값 가격', value=f'US ${df_select["price"].median()}')

# 생산년도별 히스토그램
fig_year_hist = px.histogram(df_select, x='make-year', y='price', title='생산년도별 가격 분포',
                             labels={'make-year': '생산년도', 'price': '가격'}, nbins=20, color='make-year')
st.plotly_chart(fig_year_hist)

# Streamlit 스타일 숨기기 및 푸터 추가
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    footer:after{
        content: 'Created by Samson Afolabi';
        visibility: visible;
        position: relative;
        right: 115px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)