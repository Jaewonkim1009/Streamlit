import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd

# 한국거래소 상장종목 전체 가져오기
df = fdr.StockListing("KRX")

# 5건을 화면에 출력하기
st.dataframe(df.head())

# 조회 시작 일을 입력받기
date = st.date_input(
    '조회 시작일을 선택 해주세요.',
    datetime.datetime(2022, 1, 1) # defalut 값을 2022.1.1로 입력
)

# 종목 코드를 입력받아 검색하기
code = st.text_input(
    '종목 코드',
    value='',
    placeholder='종목코드를 입력 해 주세요'
)

# 코드를 입력 했을 때 코드에 대한 정보 나타내기
if code and date: # code와 date에 값이 있을 때
    df = fdr.DataReader(code, date)
    # 날짜를 기준으로 종가를 가져옴
    data = df.sort_index(ascending=True).loc[:, 'Close']
    # 값을 선 모양 차트로 나타냄
    st.line_chart(data)