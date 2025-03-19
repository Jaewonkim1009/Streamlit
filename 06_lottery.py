import streamlit as st
import random
from datetime import datetime

st.title(':sunrise:로또 생성기:sunrise:')
my_button = st.button('로또를 생성 해 주세요')

def lottery_number():
    return sorted(random.sample(range(1,46),6))

if my_button:
    lotto_set = [lottery_number() for _ in range(5)]
    for i in range(5):
        st.write(f'{i + 1}. :red[행운의 번호]: :green[{lotto_set[i]}]')
    st.write('번호 생성 시간 : ',datetime.now().strftime('%Y-%m-%d %H:%m'))