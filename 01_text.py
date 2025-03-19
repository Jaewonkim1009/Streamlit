import streamlit as st

# title 적용 예시
st.title ('스트림릿 텍스트 적용하기')

# emoji : https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title('스마일 :sunglasses:')
st.link_button(':sunglasses:', 'https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/')

# header 적용
st.header('헤더를 입력 할 수 있어요 :sparkles:')

# Subheader 적용
st.subheader('이것은 subheader 입니다.')

# 캡션 적용
st.caption('캡션을 한 번 넣어 봤습니다.')

# 코드 표시
sample_code = '''
def function():
    print('Hello, world!')
'''
st.code(sample_code, language='python')

# st.write로 위의 모든 것을 사용 할 수 있다.
st.write('일반적인 텍스트')
# 강조 할 텍스트를 **텍스트**로 감싸 준다
st.write('이것은 마크다운 **강조** 입니다')
# 글자 앞에 #을 입력하면 갯수에따라 글자의 크기가 줄어든다.
st.write('# 제목1')
st.write('## 제목2')
st.write('### 제목3')
st.write('#### 제목4')
st.write('##### 제목5')
st.write('###### 제목6')
st.write('####### 제목7') # # 6개까지 적용 된다

# 일반 텍스트
st.text('일반적인 텍스트를 입력해 보았습니다')

# 마크다운 문법 지원
# 텍스트를 쉽게 포맷팅 할 수 있는 문법이다.

st.markdown('streamlit은 **마크다운 문법을 지원**합니다.')


# 컬러코드 : blue, green, orange, red, violet
st.markdown('텍스트의 색상을 :green[초록색]으로, 그리고 **:blue[파란색]** 볼드체로 설정할 수 있습니다.')
st.markdown(':orange[\sqrt{x^2+y^2}=1$] 와 같이 latex 문법의 수식 표현도 가능합니다. :pencil:')

# LaTeX 수식 지원
st.latex(r'\sqrt{x^2+y^2}=1')

test1 = 1
test2 = [1, 2, 3]
test3 = {'이름': '홍길동', '나이': 25}

st.write('test1: ', test1)
st.write('test2: ', test2)
st.write('test3: ', test3)