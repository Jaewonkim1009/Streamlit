import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import mysql.connector
from mysql.connector import Error
import pymysql

# 나의 MySQL 연결 정보
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'tabledb'
DB_USER = 'root'
DB_PASS = '1234'
DB_TABLE = 'cars' # 테이블 이름

# CSV 파일 경로
CSV_FILE_PATH = './data/cars.csv'

# pip install mysql-connector-python
# import mysql.connector

# 데이터베이스 연결 함수 생성
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
    
# 테이블 만들기 함수 생성
def create_table(connection):
    try:
        # 객체 생성
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            foreign_local_used VARCHAR(20),
            color VARCHAR(20),
            wheel_drive INT,
            automation VARCHAR(20),
            seat_make VARCHAR(20),
            price BIGINT,
            description VARCHAR(100),
            make_year INT,
            manufacturer VARCHAR(50)
        );
        """
        # 쿼리문 전달
        cursor.execute(create_table_query)
        # 적용
        connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f'테이블 생성 중 오류 발생: {e}')
    finally:
        cursor.close()

# 데이터 입력 함수 생성
def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        # DB_TABLE에 있는 아래 내용들을 VALUES(%s)로 문자열들을 읽어온다
        insert_query = f"""
        INSERT INTO {DB_TABLE} (
            foreign_local_used, color, wheel_drive, automation, seat_make, 
            price, description, make_year, manufacturer
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # 판다스의 데이터프레임을 numpy 배열로 만듦
        # 배열에서 하나씩 (1 record) 꺼내어 튜플로 변환하여 리스트로 반환
        data = [tuple(row) for row in df.values]
        # insert_query에 대한 것을 data가 많을때 data 만큼 처리
        cursor.executemany(insert_query, data)
        # 적용
        connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}' 테이블에 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()

# 함수 호출을 하는 역할
def main():
    # CSV 파일 읽기 (index_col=0)으로 첫 번째 열을 인덱스로 설정)
    try:
        df = pd.read_csv(CSV_FILE_PATH, index_col=0)
        print('CSV 파일을 성공적으로 읽었습니다.')
    except FileNotFoundError:
        print(f"파일 '{CSV_FILE_PATH}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 읽기 중 오류 발생 : {e}")
        return

    # 열 이름 조정 (공백이나 다른 문자를 _로 변경)
    excepted_columns = [
        'foreign_local_used', 'color', 'wheel_drive', 'automation',
        'seat_make', 'price', 'description', 'make_year', 'manufacturer'
    ]
    df.columns = excepted_columns # 열 이름 매핑

    # 데이터베이스 연결
    connection = create_database_connection()
    if connection is None:
        return
    
    # 테이블 생성
    create_table(connection)

    # 데이터 삽입
    insert_data(connection, df)
    
    # 데이터 확인
    cars_data = """
    select * from tabledb.cars" 
    """

    # 연결 종료
    if connection.is_connected():
        connection.close()
        print("MySQL 연결이 종료 되었습니다.")

if __name__ == '__main__':
    main()

# SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# 데이터 가져오기 함수
def fetch_data():
    query = 'select * from tabledb.cars;'
    df = pd.read_sql(query, engine) # SQLAlchemy 사용
    return df

# Streamlit 앱
st.title('MySQL Cars 확인하기')
if st.button('데이터 확인'):
    df = fetch_data()
    st.dataframe(df) # 데이터프레임으로 데이터 출력