import streamlit as st

# Session state 사용 전
# 클릭을 아무리 해도 트리거가 True로 변해 조건이 끝나 1에서 늘어나지 않는다
counter = 0

button = st.button('Click!')
if button:
    counter += 1

st.write(f'버튼을 {counter}번 클릭 하셨습니다.')

# Session state 사용 후
if "counter" not in st.session_state:
    st.session_state.counter = 0

button2 = st.button('클릭!')
if button2:
    st.session_state.counter += 1
st.write(f'버튼을{st.session_state.counter}번 클릭 하셨습니다.')