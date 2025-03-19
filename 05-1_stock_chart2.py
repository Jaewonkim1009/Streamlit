import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd

# 한국거래소 상장종목 전체 가져오기
df = fdr.StockListing("KRX")

# 제목 입력
st.title('종목 차트 검색')

# 조회 시작 일을 입력받기
with st.sidebar:
    date = st.date_input(
    '조회 시작일을 선택 해주세요.',
    datetime.datetime(2022, 1, 1) # default 값을 2022.1.1로 입력
)
# 종목 코드를 입력받아 검색하기
with st.sidebar:
    code = st.text_input(
    '종목 코드',
    value = '005930',
    placeholder='종목코드를 입력 해 주세요'
)

# 코드를 입력 했을 때 코드에 대한 정보 나타내기
tab1, tab2 = st.tabs(['차트', '데이터'])
with tab1:
    if code and date: # code와 date에 값이 있을 때
        df = fdr.DataReader(code, date)
        # 날짜를 기준으로 종가를 가져옴
        data = df.sort_index(ascending=True).loc[:, 'Close']
        # 값을 선 모양 차트로 나타냄
        st.line_chart(data)

with tab2:
    if code and date: # code와 date에 값이 있을 때
        df = fdr.DataReader(code, date)
        # df.head()로 5개만 출력
        st.write(df.head())

# 다른 쪽에 동일한 것을 나타내고 싶을 때 st.expander()
with st.expander('칼럼 설명'):
    st.markdown('''
    - Open: 시가
    - High: 고가
    - Low: 저가
    - Close: 종가
    - Volumn: 거래량
    - Adj Close: 수정 종가
    '''
)