import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

재무 = pd.read_excel('섹터재무.xlsx', sheet_name= '재무')

시가총액 = pd.read_excel('섹터재무.xlsx', sheet_name= '일별시가총액')


경제지표 = pd.read_excel('섹터재무.xlsx', sheet_name= '일별경제지표')


가격 = pd.read_excel('섹터재무.xlsx', sheet_name= 'ETF일가격')


# 'IT' 문자열을 '정보기술'로 치환
가격['Symbol Name'] = 가격['Symbol Name'].str.replace('IT', '정보기술')

# 첫 공백 이전의 단어와 공백을 제거 (정규식을 사용)
가격['Symbol Name'] = 가격['Symbol Name'].str.replace(r'^\S+\s', '', regex=True)

# 첫 공백 이전의 단어와 공백을 제거 (정규식을 사용)
재무['Symbol Name'] = 재무['Symbol Name'].str.replace(r'^\S+\s', '', regex=True)
# 첫 공백 이전의 단어와 공백을 제거 (정규식을 사용)
시가총액['Symbol Name'] = 시가총액['Symbol Name'].str.replace(r'^\S+\s', '', regex=True)

가격.drop(가격.columns[[0, 2, 3, 5]], axis=1, inplace=True)
시가총액.drop(시가총액.columns[[0, 2, 3, 5]], axis=1, inplace=True)
재무.drop(재무.columns[[0, 2, 3, 5]], axis=1, inplace=True)
경제지표.drop(경제지표.columns[[0, 1, 2, 3,5]], axis=1, inplace=True)


가격 = 가격.melt(id_vars=가격.columns[:2], var_name="time", value_name="Value")
가격 = pd.pivot_table(가격,
                       index=['Symbol Name', 'time'],
                       columns='Item Name',
                       values='Value',
                       aggfunc='first')

시가총액 = 시가총액.melt(id_vars=시가총액.columns[:2], var_name="time", value_name="Value")
시가총액 = pd.pivot_table(시가총액,
                           index=['Symbol Name', 'time'],
                           columns='Item Name',
                           values='Value',
                           aggfunc='first')

시가총액.iloc[:, 0] = 시가총액.iloc[:, 0].mask((시가총액.iloc[:, 0].isna() | (시가총액.iloc[:, 0] == 0)) & (시가총액.iloc[:, 1] >= 1), 시가총액.iloc[:, 1])
시가총액.drop(시가총액.columns[1], axis=1, inplace=True)

경제지표 = 경제지표.melt(id_vars=경제지표.columns[:1], var_name="time", value_name="Value")

경제지표2 = pd.pivot_table(경제지표,
                           index='time',
                           columns='Item Name',
                           values='Value',
                           aggfunc='first')


재무 = 재무.melt(id_vars=재무.columns[:2], var_name="time", value_name="Value")
재무 = pd.pivot_table(재무,
                       index=['Symbol Name', 'time'],
                       columns='Item Name',
                       values='Value',
                       aggfunc='first')


# 1️⃣ 인덱스를 컬럼으로 변환
재무 = 재무.reset_index()

# 2️⃣ 'time' 컬럼을 날짜 형식으로 변환 후 하루 증가
재무['time'] = pd.to_datetime(재무['time']) + pd.Timedelta(days=1)

# 3️⃣ 특정 월에 따라 추가적인 월 증가
재무['time'] = 재무['time'].apply(lambda x: 
    x + relativedelta(months=2) if x.month in [4, 7, 10] else 
    x + relativedelta(months=3) if x.month == 1 else 
    x
)

# 4️⃣ 다시 MultiIndex로 설정
재무 = 재무.set_index(['Symbol Name', 'time'])

# 1️⃣ 날짜 범위를 데이터프레임으로 변환
dates_df = pd.DataFrame({'time': pd.date_range(start="2004-04-01", end="2025-01-31", freq="D")})

# 2️⃣ 'Symbol Name'과 모든 날짜를 조합하여 데이터 확장
symbol_names = 재무.index.get_level_values('Symbol Name').unique()  # 기존 종목 리스트
expanded_dates = dates_df.merge(pd.DataFrame({'Symbol Name': symbol_names}), how='cross')  # 모든 날짜 × Symbol Name

# 3️⃣ 'Symbol Name'과 'time'을 기준으로 재무 데이터 병합
재무 = expanded_dates.merge(재무, on=['Symbol Name', 'time'], how='left')
    
재무 = 재무.set_index(['Symbol Name', 'time'])

# 심볼(Symbol Name)별로 NaN을 이전 값으로 채우기
재무 = 재무.groupby('Symbol Name').ffill()


# 가격과 시가총액을 'Symbol Name'과 'time'을 기준으로 outer join
통합데이터 = 가격.merge(시가총액, on=['Symbol Name', 'time'], how='outer')

# 1️⃣ 'Symbol Name'과 'time'을 기준으로 재무 데이터 병합
통합데이터 = 통합데이터.merge(재무, on=['Symbol Name', 'time'], how='left')



# 1️⃣ '통합데이터'에서 'Symbol Name' 리스트 추출
symbol_names = 통합데이터.index.get_level_values('Symbol Name').unique()

# 'time'이 인덱스라면 일반 컬럼으로 변환
경제지표2 = 경제지표2.reset_index()

# 2️⃣ '경제지표2'에 모든 Symbol Name을 추가하여 MultiIndex 생성
경제지표2_multi = 경제지표2.merge(
    pd.DataFrame({'Symbol Name': symbol_names}), how='cross'
).set_index(['Symbol Name', 'time'])

경제지표2_multi = 경제지표2_multi.groupby('Symbol Name').ffill()



# 3️⃣ '통합데이터'와 '경제지표2' 병합 (MultiIndex 유지)
통합데이터 = 통합데이터.merge(경제지표2_multi, on=['Symbol Name', 'time'], how='left')

통합데이터 = 통합데이터[['수정주가 (현금배당반영)(원)', '시가총액 (KRX)(원)', '*총금융부채(원)', '단기금융부채(원)',
       '단기금융자산(원)', '당기순이익(원)', '당좌자산(원)', '매출액(원)', '매출채권(원)', '무형자산(원)',
       '비유동부채(원)', '비유동자산(원)', '영업이익(원)', '유동부채(원)', '유동자산(원)', '유형자산(원)', '장기금융부채(원)', '재고자산(원)', 
       '총부채(원)', '총자본(원)', '총자산(원)',
       '총포괄이익(원)', '현금및현금성자산(원)',
       'DDR3 4Gb 512Mx8 eTT(USD)',
       'DDR4 16G (2G*8) eTT MHZ(USD)', '국채금리_미국국채(10년)(%)', '국채금리_미국국채(1년)(%)',
       '금(선물)($/ounce)', '니켈(선물)($/ton)', '시장금리:CD유통수익률(91)(%)',
       '시장금리:국고10년(%)', '시장금리:회사채(무보증3년AA-)(%)', '시장평균_미국(달러)(통화대원)',
       '시장평균_일본(100엔)((100)통화대원)', '시장평균_중국(위안)(통화대원)',
       '주요상품선물_DUBAI(ASIA1M)($/bbl)', '주요상품선물_소맥(최근월물)(￠/bu)',
       '주요상품선물_전기동(선물)($/ton)']]


통합데이터[통합데이터.columns.difference(["수정주가 (현금배당반영)(원)"])] = 통합데이터[통합데이터.columns.difference(["수정주가 (현금배당반영)(원)"])].fillna(0)

# 한글 → 영어 매핑 딕셔너리
sector_translation = {
    '건설': 'Construction',
    '경기소비재': 'Consumer Discretionary',
    '기계장비': 'Machinery Equipment',
    '미디어&엔터테인먼트': 'Media Entertainment',
    '반도체': 'Semiconductors',
    '방송통신': 'Broadcasting Telecommunications',
    '보험': 'Insurance',
    '에너지화학': 'Energy Chemicals',
    '운송': 'Transportation',
    '은행': 'Banking',
    '자동차': 'Automobile',
    '정보기술': 'Information Technology',
    '증권': 'Securities',
    '철강': 'Steel',
    '필수소비재': 'Consumer Staples',
    '헬스케어': 'Healthcare'
}

# 인덱스 값 변경
통합데이터 = 통합데이터.rename(index=lambda x: sector_translation.get(x, x), level='Symbol Name')



통합데이터 = 통합데이터.query("time >= @pd.Timestamp('2014-04-02')")
통합데이터 = 통합데이터.query("time <= @pd.Timestamp('2025-01-31')")

# "수정주가 (현금배당반영)(원)" 열에 NaN이 포함된 Symbol Name을 찾고, 해당 Symbol Name의 모든 행 삭제

# 처리할 열들의 기본 이름 리스트
cols = ['당기순이익', '영업이익', '총포괄이익']

for col in cols:
    # 음수 값들을 새로운 열에 저장
    통합데이터[f'{col}(-)'] = 통합데이터[f'{col}(원)'].where(통합데이터[f'{col}(원)'] < 0)
    
    # 음수 값들을 양수로 변환하고 NA를 0으로 채움
    통합데이터[f'{col}(-)'] = 0 - 통합데이터[f'{col}(-)'].fillna(0)
    
    # 원래 열에서는 음수 값을 0으로 대체
    통합데이터[f'{col}(원)'] = 통합데이터[f'{col}(원)'].where(통합데이터[f'{col}(원)'] >= 0, 0)


cols = 통합데이터.columns.difference(["수정주가 (현금배당반영)(원)"])  # 변환할 열 선택
통합데이터[cols] = np.where(통합데이터[cols] > 0, np.log(통합데이터[cols]), 0)

통합데이터1 = 통합데이터.drop(index=통합데이터.index.get_level_values("Symbol Name")[통합데이터["수정주가 (현금배당반영)(원)"].isna()].unique())

통합데이터2 = 통합데이터.query("time >= @pd.Timestamp('2018-04-01')")



# 'time' 인덱스가 datetime 형식이어야 함
월별통합데이터1 = 통합데이터1.groupby('Symbol Name').resample('MS', level='time').first()

월별통합데이터2 = 통합데이터2.groupby('Symbol Name').resample('MS', level='time').first()




# 인덱스를 컬럼으로 변환하여 저장
통합데이터1.reset_index().to_excel('통합데이터1.xlsx', index=False)
월별통합데이터1.reset_index().to_excel('월별통합데이터1.xlsx', index=False)

통합데이터2.reset_index().to_excel('통합데이터2.xlsx', index=False)
월별통합데이터2.reset_index().to_excel('월별통합데이터2.xlsx', index=False)