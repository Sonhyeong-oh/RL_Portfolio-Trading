'''
!sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
!git clone https://github.com/AI4Finance-Foundation/FinRL.git
!git clone https://github.com/AI4Finance-Foundation/FinRL-Tutorials.git
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
!pip install swig
!pip install wrds
!pip install pyportfolioopt
!pip install finrl
!pip install quantstats
!pip install torch_geometric
!pip install optuna
!pip install torch
!pip install gym
!pip install openpyxl

!cd FinRL
!pip install .
'''


import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import torch

import numpy as np
import pandas as pd
import optuna
import joblib
import quantstats as qs
from sklearn.preprocessing import MaxAbsScaler
import sys
#sys.path.append('/content/FinRL/finrl/meta/preprocessor')
#sys.path.append('/content/new_algorithms.py')
#sys.path.append('/content/new_models.py')
#from new_algorithms import PolicyGradient
#from new_models import DRLAgent
# from google.colab import files
# sys.path.append('/content/')
# import architectures as ac

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE







# ----------------------------
# 1. 데이터 불러오기 및 전처리
# ----------------------------

# 모델 학습에 사용할 장치 선택
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 데이터 불러오기

df_portfolio = pd.read_excel('통합데이터1.xlsx')
#df_portfolio = pd.read_excel('통합데이터1.xlsx')

# Symbol(회사 코드) 별로 그룹화
df_portfolio.groupby("Symbol Name").count()

# # GroupByScaler 수행 시 '날짜' 타입이 datetime이면 오류 발생, object로 변경
# data["time"] = data["time"].astype(object)
# # 수정주가 데이터가 0 (결측치)인 데이터 제거
# data = data.loc[data["수정주가 (현금배당반영)(원)"] != 0]

# GroupByScaler = Symbol을 기준(by = "Symbol")으로 data의 그룹화 후 처리(scaler) 진행
# MaxAbsScaler = 주어진 데이터 값을 [-1, 1]로 변환, 최댓값의 절댓값을 기준으로 변환
#portfolio_norm_df = pp.GroupByScaler(by="Symbol Name", scaler=MaxAbsScaler).fit_transform(data)
# portfolio_norm_df

# portfolio_norm_df에서 해당 열을 추출하여 df_portfolio에 할당
# df_portfolio = data[["Symbol Name", "time", "수정주가 (현금배당반영)(원)", "시가총액 (KRX)(원)", "*총금융부채(원)",
#                                   "단기금융부채(원)", "단기금융자산(원)", "당기순이익(원)", "당좌자산(원)", "매출액(원)",
#                                   "매출채권(원)", "무형자산(원)", "비유동부채(원)", "비유동자산(원)", "영업이익(원)", "유동부채(원)",
#                                   "유동자산(원)", "유형자산(원)", "장기금융부채(원)", "재고자산(원)", "총부채(원)", "총자본(원)", "총자산(원)",
#                                   "총포괄이익(원)", "현금및현금성자산(원)", "DDR3 4Gb 512Mx8 eTT(USD)", "DDR4 16G (2G*8) eTT MHZ(USD)",
#                                   "국채금리_미국국채(10년)(%)", "국채금리_미국국채(1년)(%)", "금(선물)($/ounce)", "니켈(선물)($/ton)",
#                                   "시장금리:CD유통수익률(91)(%)", "시장금리:국고10년(%)", "시장금리:회사채(무보증3년AA-)(%)",
#                                   "시장평균_미국(달러)(통화대원)", "시장평균_일본(100엔)((100)통화대원)", "시장평균_중국(위안)(통화대원)",
#                                   "주요상품선물_DUBAI(ASIA1M)($/bbl)", "주요상품선물_소맥(최근월물)(￠/bu)", "주요상품선물_전기동(선물)($/ton)",
#                                   "당기순이익(-)", "영업이익(-)", "총포괄이익(-)"]]
# # 날짜에 따른 분할을 수행하기 위해 '날짜'를 다시 datetime 타입으로 변환
df_portfolio["time"] = df_portfolio["time"].astype('datetime64[ns]')



sptdate = "2021-04-01"

# 학습 데이터와 테스트 데이터를 date 기준에 따라 분할
df_portfolio_train = df_portfolio[(df_portfolio["time"] < sptdate)]
df_portfolio_test = df_portfolio[(df_portfolio["time"] >= sptdate)]

'''
# 학습 데이터와 테스트 데이터를 date 기준으로 분할
df_portfolio_train = df_portfolio[df_portfolio["time"] < sptdate].copy()
df_portfolio_test = df_portfolio[df_portfolio["time"] >= sptdate].copy()

# 첫 번째, 두 번째 열 제외한 나머지 열 선택
train_subset = df_portfolio_train.iloc[:, 2:]
test_subset = df_portfolio_test.iloc[:, 2:]

# 각 열의 최댓값 계산 (0 제외)
train_max_values = train_subset.replace(0, np.nan).max()
test_max_values = test_subset.replace(0, np.nan).max()

# 값이 0이 아닌 경우에만 나누기
df_portfolio_train.iloc[:, 2:] = train_subset.div(train_max_values).fillna(0)
df_portfolio_test.iloc[:, 2:] = test_subset.div(test_max_values).fillna(0)
'''
# 2017년 전 데이터가 존재하는 symbol name을 train_list에 저장
# train_list = df_portfolio[df_portfolio['time'] < '2017-01-01']['Symbol Name'].unique()
# train_list에 해당하는 데이터만 추출하여 train_data로 저장
# df_portfolio_train = df_portfolio[df_portfolio['Symbol Name'].isin(train_list)]

# # train, test 데이터셋 분리
# df_portfolio_train = df_portfolio_train[(df_portfolio_train["time"] >= "2010-01-01") & (df_portfolio_train["time"] < "2019-01-01")]
# df_portfolio_test = df_portfolio[(df_portfolio["time"] >= "2019-01-01") & (df_portfolio["time"] < "2025-01-01")]




# df_portfolio_train에서 각 그룹별 시작값으로 정규화
df_portfolio_train["수정주가 (현금배당반영)(원)"] = df_portfolio_train.groupby("Symbol Name")["수정주가 (현금배당반영)(원)"] \
    .transform(lambda x: x / x.iloc[0])

# df_portfolio_test에서도 동일하게 각 그룹별 시작값으로 정규화
df_portfolio_test["수정주가 (현금배당반영)(원)"] = df_portfolio_test.groupby("Symbol Name")["수정주가 (현금배당반영)(원)"] \
    .transform(lambda x: x / x.iloc[0])


del df_portfolio


torch.set_num_threads(12)


# ----------------------
# 2. 모델 학습 환경 설정
# ----------------------
features = ["수정주가 (현금배당반영)(원)", "시가총액 (KRX)(원)", "*총금융부채(원)",
                                   "단기금융부채(원)", "단기금융자산(원)", "당기순이익(원)", "당좌자산(원)", "매출액(원)",
                                   "매출채권(원)", "무형자산(원)", "비유동부채(원)", "비유동자산(원)", "영업이익(원)", "유동부채(원)",
                                   "유동자산(원)", "유형자산(원)", "장기금융부채(원)", "재고자산(원)", "총부채(원)", "총자본(원)", "총자산(원)",
                                   "총포괄이익(원)", "현금및현금성자산(원)", "DDR3 4Gb 512Mx8 eTT(USD)", "DDR4 16G (2G*8) eTT MHZ(USD)",
                                   "국채금리_미국국채(10년)(%)", "국채금리_미국국채(1년)(%)", "금(선물)($/ounce)", "니켈(선물)($/ton)",
                                   "시장금리:CD유통수익률(91)(%)", "시장금리:국고10년(%)", "시장금리:회사채(무보증3년AA-)(%)",
                                   "시장평균_미국(달러)(통화대원)", "시장평균_일본(100엔)((100)통화대원)", "시장평균_중국(위안)(통화대원)",
                                   "주요상품선물_DUBAI(ASIA1M)($/bbl)", "주요상품선물_소맥(최근월물)(￠/bu)", "주요상품선물_전기동(선물)($/ton)",
                                   "당기순이익(-)", "영업이익(-)", "총포괄이익(-)"]

# %%


timewin = 50
ksize = 3
fee = 0.0012
normalize = None
mid_features = 30
final_features = 20
initial_features = len(features)

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_model='trf',
        comission_fee_pct=fee,
        time_window=timewin,
        features= features,
        tic_column="Symbol Name",
        time_column="time",
        valuation_feature="수정주가 (현금배당반영)(원)",
        normalize_df=None,
        cwd= '/content'
    )




environment_test = PortfolioOptimizationEnv(
    df_portfolio_test,
    initial_amount=100000,
    comission_fee_pct=fee,
    time_window=timewin,
    features=features,
    tic_column="Symbol Name",
    time_column="time",
    valuation_feature="수정주가 (현금배당반영)(원)",
    normalize_df=None,
    cwd='/content'

)




# 강화학습 모델 파라미터 설정
model_kwargs = {
    "lr": 0.000003,
    "policy": EIIE,
    "batch_size": 128,
    "action_noise": 0,
   # "validation_env":environment_test

  }

# 정책 신경망(EIIE)의 파라미터 설정
policy_kwargs = {
    "k_size": ksize,
    "time_window": timewin,
    "initial_features" : initial_features,
    'conv_mid_features' : mid_features,
    'conv_final_features' : final_features
  }

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)

# 설정한 모델 훈련
'''
model = 위에서 지정한 강화학습 모델
episodes = 학습 횟수 (epoch와 같은 의미)
'''
DRLAgent.train_model(model, episodes=70)

# 훈련된 정책 네트워크를 policy_EIIE.pt로 저장
torch.save(model.train_policy.state_dict(), "policy_EIIE.pt")



# ----------------------------------
# 4. 테스트 환경 설정 및 테스트 진행
# ----------------------------------


# 정책 신경망 설정

policy = EIIE(time_window=timewin,
              device=device,
              initial_features=initial_features,
              k_size = ksize,
              conv_mid_features =mid_features,
             conv_final_features = final_features
              )

# 저장되어있는 EIIE의 파라미터를 불러옴
policy.load_state_dict(torch.load("policy_EIIE.pt"))

# 모델의 성능 평가 후 최종 가치를 EIIE result에 저장


EIIE_results = {
    "train": {},
    "test": {},
}

DRLAgent.DRL_validation(model, environment_train, policy=None)
EIIE_results["train"]["value"] = environment_train._asset_memory["final"]
EIIE_results["train"]["weights"] = environment_train._final_weights

DRLAgent.DRL_validation(model, environment_test,
                        policy=policy,
                        online_training_period= 100000)
EIIE_results["test"]["value"] = environment_test._asset_memory["final"]
EIIE_results["test"]["weights"] = environment_test._final_weights




# ---------------------------------------------------------------------
# 5. 강화학습 모델과의 성능 비교를 위해 Buy and hold 전략으로 학습 진행
# ---------------------------------------------------------------------

UBAH_results = {
    "train": {},
    "test": {}
}

# train 데이터의 회사 갯수를 TRAIN_PORTFOLIO_SIZE에 저장
train_symbol_list = list(set(df_portfolio_train['Symbol Name']))
TRAIN_PORTFOLIO_SIZE = len(train_symbol_list)

# test 데이터의 회사 갯수를 TEST_PORTFOLIO_SIZE에 저장
test_symbol_list = list(set(df_portfolio_test['Symbol Name']))
TEST_PORTFOLIO_SIZE = len(test_symbol_list)

# train 환경
'''
terminated = 학습 종료 시점을 알기 위한 변수
environment.reset() = 학습 환경 초기화
action = 포트폴리오에 대한 자산 배분 전략
        [0]: 첫 번째 자산(예: 현금)의 비율이 0으로 설정, 현금 자산에 대한 투자는 하지 않겠다는 뜻
        [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE: 나머지 자산에 대해서는 동일한 비율로 분배
        PORTFOLIO_SIZE는 자산의 수를 나타내므로, 각 자산에 대해 동일한 비율인 1/PORTFOLIO_SIZE만큼 배분
UBAH_results... = UBAH_results 딕셔너리에 최종 가치를 저장장
'''
terminated = False
environment_train.reset()

while not terminated:
    action = [0] + [1/TRAIN_PORTFOLIO_SIZE] * TRAIN_PORTFOLIO_SIZE
    _, _, terminated, _ = environment_train.step(action)
UBAH_results["train"]["value"] = environment_train._asset_memory["final"]
UBAH_results["train"]["weights"] = environment_train._final_weights

# 테스트 환경
terminated = False
environment_test.reset()
while not terminated:
    action = [0] + [1/TEST_PORTFOLIO_SIZE] * TEST_PORTFOLIO_SIZE
    _, _, terminated, _ = environment_test.step(action)
UBAH_results["test"]["value"] = environment_test._asset_memory["final"]
UBAH_results["test"]["weights"] = environment_test._final_weights



import matplotlib
matplotlib.use('Agg')
# %matplotlib inline
import matplotlib.pyplot as plt

# 중복 제거 후 날짜를 가져오고, timewin - 1 만큼 날짜를 조정
train_time = np.sort(df_portfolio_train["time"].unique())  # 시간 정렬
train_time = train_time[timewin - 1:]

test_time = np.sort(df_portfolio_test["time"].unique())  # 시간 정렬
test_time = test_time[timewin - 1:]

# 결과 데이터를 DataFrame으로 변환하여 올바른 인덱스 설정
train_buy_hold_df = pd.DataFrame({"value": UBAH_results["train"]["value"]}, index=train_time)
train_eiie_df = pd.DataFrame({"value": EIIE_results["train"]["value"]}, index=train_time)

test_buy_hold_df = pd.DataFrame({"value": UBAH_results["test"]["value"]}, index=test_time)
test_eiie_df = pd.DataFrame({"value": EIIE_results["test"]["value"]}, index=test_time)

# ---------
# 6. 시각화 (타임 인덱스 적용)
# ---------

# 학습 환경(2011년)에서의 UBAH (Buy and Hold) 전략과 EIIE 전략의 최종 가치 비교
plt.figure(figsize=(10, 5))
plt.plot(train_buy_hold_df.index, train_buy_hold_df["value"], label="Buy and Hold")
plt.plot(train_eiie_df.index, train_eiie_df["value"], label="EIIE")

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Performance in Training Period")
plt.legend()
plt.xticks(rotation=45)  # x축 라벨 회전
plt.grid(True)
plt.show()

# 테스트 환경(2012년)에서의 UBAH (Buy and Hold) 전략과 EIIE 전략의 최종 가치 비교
plt.figure(figsize=(10, 5))
plt.plot(test_buy_hold_df.index, test_buy_hold_df["value"], label="Buy and Hold")
plt.plot(test_eiie_df.index, test_eiie_df["value"], label="EIIE")

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Performance in Test Period")
plt.legend()
plt.xticks(rotation=45)  # x축 라벨 회전
plt.grid(True)
plt.show()





import quantstats as qt

# 예시: df_portfolio_test는 'time' 컬럼에 날짜 정보를 가지고 있다고 가정합니다.
# 그리고 ubah_values와 eiiE_values는 각각 UBAH와 EIIE 테스트 결과로 나온 포트폴리오 가치 리스트입니다.


# 중복 제거 후 날짜를 가져오고, timewin + 1 만큼 날짜를 조정
train_time = df_portfolio_train["time"].unique()
train_time = train_time[timewin -1 :]


# 예시: UBAH와 EIIE의 테스트 결과가 리스트 형태라고 가정합니다.
ubah_values = UBAH_results['train']['value']
eiiE_values = EIIE_results['train']['value']

# 데이터프레임의 "time" 컬럼을 인덱스로 사용하여 Series 생성
ubah_price = pd.Series(ubah_values, index=train_time)
eiiE_price = pd.Series(eiiE_values, index=train_time)

# 로그 차분을 사용해 로그 수익률 계산
ubah_return = np.log(ubah_price).diff().dropna()
eiiE_return = np.log(eiiE_price).diff().dropna()

# Quantstats를 사용하여 Sharpe Ratio 계산
sharpe_ubah = qt.stats.sharpe(ubah_return)
sharpe_eiiE = qt.stats.sharpe(eiiE_return)

print("UBAH Sharpe Ratio:", sharpe_ubah)
print("EIIE Sharpe Ratio:", sharpe_eiiE)






# 중복 제거 후 날짜를 가져오고, timewin + 1 만큼 날짜를 조정
test_time = df_portfolio_test["time"].unique()
test_time = test_time[timewin -1 :]


# 예시: UBAH와 EIIE의 테스트 결과가 리스트 형태라고 가정합니다.
ubah_testvalues = UBAH_results['test']['value']
eiiE_tsetvalues = EIIE_results['test']['value']

# 데이터프레임의 "time" 컬럼을 인덱스로 사용하여 Series 생성
ubah_test_price = pd.Series(ubah_testvalues, index=test_time)
eiiE_test_price = pd.Series(eiiE_tsetvalues, index=test_time)

# 로그 차분을 사용해 로그 수익률 계산
ubah_test_return = np.log(ubah_test_price).diff().dropna()
eiiE_test_return = np.log(eiiE_test_price).diff().dropna()

# Quantstats를 사용하여 Sharpe Ratio 계산
sharpe_ubah_test = qt.stats.sharpe(ubah_test_return)
sharpe_eiiE_test = qt.stats.sharpe(eiiE_test_return)

print("UBAH Sharpe Ratio test:", sharpe_ubah_test)
print("EIIE Sharpe Ratio test:", sharpe_eiiE_test)



test_buy_hold_dfw = pd.DataFrame({"value": UBAH_results["test"]["weights"]}, index=test_time)
test_eiie_dfw = pd.DataFrame({"value": EIIE_results["test"]["weights"]}, index=test_time)


import pandas as pd

weights_list = EIIE_results["train"]["weights"]

# 각 배열을 DataFrame으로 변환한 후, axis=1로 결합하면 (n, m) 형태의 DataFrame이 됩니다.
df = pd.concat([pd.DataFrame(arr) for arr in weights_list], axis=1)

# %%



timewin = 50
ksize = 3
fee = 0.0012
normalize = None
mid_features = 30
final_features = 20
initial_features = len(features)

environment_train = PortfolioOptimizationEnv(
        df_portfolio_train,
        initial_amount=100000,
        comission_fee_model='trf',
        comission_fee_pct=fee,
        time_window=timewin,
        features= features,
        tic_column="Symbol Name",
        time_column="time",
        valuation_feature="수정주가 (현금배당반영)(원)",
        normalize_df=None,
        cwd= '/content'
    )




environment_test = PortfolioOptimizationEnv(
    df_portfolio_test,
    initial_amount=100000,
    comission_fee_pct=fee,
    time_window=timewin,
    features=features,
    tic_column="Symbol Name",
    time_column="time",
    valuation_feature="수정주가 (현금배당반영)(원)",
    normalize_df=None,
    cwd='/content'

)




# 강화학습 모델 파라미터 설정
model_kwargs = {
    "lr": 0.00001,
    "policy": EIIE,
    "batch_size": 128,
    "action_noise": 0,
   # "validation_env":environment_test

  }

# 정책 신경망(EIIE)의 파라미터 설정
policy_kwargs = {
    "k_size": ksize,
    "time_window": timewin,
    "initial_features" : initial_features,
    'conv_mid_features' : mid_features,
    'conv_final_features' : final_features
  }

model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)

# 설정한 모델 훈련
'''
model = 위에서 지정한 강화학습 모델
episodes = 학습 횟수 (epoch와 같은 의미)
'''
DRLAgent.train_model(model, episodes=10)

# 훈련된 정책 네트워크를 policy_EIIE.pt로 저장
torch.save(model.train_policy.state_dict(), "policy_EIIE.pt")



# ----------------------------------
# 4. 테스트 환경 설정 및 테스트 진행
# ----------------------------------


# 정책 신경망 설정

policy = EIIE(time_window=timewin,
              device=device,
              initial_features=initial_features,
              k_size = ksize,
              conv_mid_features =mid_features,
             conv_final_features = final_features
              )

# 저장되어있는 EIIE의 파라미터를 불러옴
policy.load_state_dict(torch.load("policy_EIIE.pt"))

# 모델의 성능 평가 후 최종 가치를 EIIE result에 저장


EIIE_results = {
    "train": {},
    "test": {},
}

DRLAgent.DRL_validation(model, environment_train, policy=None)
EIIE_results["train"]["value"] = environment_train._asset_memory["final"]
EIIE_results["train"]["weights"] = environment_train._final_weights

DRLAgent.DRL_validation(model, environment_test,
                        policy=policy,
                        online_training_period= 100000)
EIIE_results["test"]["value"] = environment_test._asset_memory["final"]
EIIE_results["test"]["weights"] = environment_test._final_weights




# ---------------------------------------------------------------------
# 5. 강화학습 모델과의 성능 비교를 위해 Buy and hold 전략으로 학습 진행
# ---------------------------------------------------------------------

UBAH_results = {
    "train": {},
    "test": {}
}

# train 데이터의 회사 갯수를 TRAIN_PORTFOLIO_SIZE에 저장
train_symbol_list = list(set(df_portfolio_train['Symbol Name']))
TRAIN_PORTFOLIO_SIZE = len(train_symbol_list)

# test 데이터의 회사 갯수를 TEST_PORTFOLIO_SIZE에 저장
test_symbol_list = list(set(df_portfolio_test['Symbol Name']))
TEST_PORTFOLIO_SIZE = len(test_symbol_list)

# train 환경
'''
terminated = 학습 종료 시점을 알기 위한 변수
environment.reset() = 학습 환경 초기화
action = 포트폴리오에 대한 자산 배분 전략
        [0]: 첫 번째 자산(예: 현금)의 비율이 0으로 설정, 현금 자산에 대한 투자는 하지 않겠다는 뜻
        [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE: 나머지 자산에 대해서는 동일한 비율로 분배
        PORTFOLIO_SIZE는 자산의 수를 나타내므로, 각 자산에 대해 동일한 비율인 1/PORTFOLIO_SIZE만큼 배분
UBAH_results... = UBAH_results 딕셔너리에 최종 가치를 저장장
'''
terminated = False
environment_train.reset()

while not terminated:
    action = [0] + [1/TRAIN_PORTFOLIO_SIZE] * TRAIN_PORTFOLIO_SIZE
    _, _, terminated, _ = environment_train.step(action)
UBAH_results["train"]["value"] = environment_train._asset_memory["final"]
UBAH_results["train"]["weights"] = environment_train._final_weights

# 테스트 환경
terminated = False
environment_test.reset()
while not terminated:
    action = [0] + [1/TEST_PORTFOLIO_SIZE] * TEST_PORTFOLIO_SIZE
    _, _, terminated, _ = environment_test.step(action)
UBAH_results["test"]["value"] = environment_test._asset_memory["final"]
UBAH_results["test"]["weights"] = environment_test._final_weights



import matplotlib
matplotlib.use('Agg')
# %matplotlib inline
import matplotlib.pyplot as plt

# 중복 제거 후 날짜를 가져오고, timewin - 1 만큼 날짜를 조정
train_time = np.sort(df_portfolio_train["time"].unique())  # 시간 정렬
train_time = train_time[timewin - 1:]

test_time = np.sort(df_portfolio_test["time"].unique())  # 시간 정렬
test_time = test_time[timewin - 1:]

# 결과 데이터를 DataFrame으로 변환하여 올바른 인덱스 설정
train_buy_hold_df = pd.DataFrame({"value": UBAH_results["train"]["value"]}, index=train_time)
train_eiie_df = pd.DataFrame({"value": EIIE_results["train"]["value"]}, index=train_time)

test_buy_hold_df = pd.DataFrame({"value": UBAH_results["test"]["value"]}, index=test_time)
test_eiie_df = pd.DataFrame({"value": EIIE_results["test"]["value"]}, index=test_time)

# ---------
# 6. 시각화 (타임 인덱스 적용)
# ---------

# 학습 환경(2011년)에서의 UBAH (Buy and Hold) 전략과 EIIE 전략의 최종 가치 비교
plt.figure(figsize=(10, 5))
plt.plot(train_buy_hold_df.index, train_buy_hold_df["value"], label="Buy and Hold")
plt.plot(train_eiie_df.index, train_eiie_df["value"], label="EIIE")

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Performance in Training Period")
plt.legend()
plt.xticks(rotation=45)  # x축 라벨 회전
plt.grid(True)
plt.show()

# 테스트 환경(2012년)에서의 UBAH (Buy and Hold) 전략과 EIIE 전략의 최종 가치 비교
plt.figure(figsize=(10, 5))
plt.plot(test_buy_hold_df.index, test_buy_hold_df["value"], label="Buy and Hold")
plt.plot(test_eiie_df.index, test_eiie_df["value"], label="EIIE")

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Performance in Test Period")
plt.legend()
plt.xticks(rotation=45)  # x축 라벨 회전
plt.grid(True)
plt.show()





import quantstats as qt

# 예시: df_portfolio_test는 'time' 컬럼에 날짜 정보를 가지고 있다고 가정합니다.
# 그리고 ubah_values와 eiiE_values는 각각 UBAH와 EIIE 테스트 결과로 나온 포트폴리오 가치 리스트입니다.


# 중복 제거 후 날짜를 가져오고, timewin + 1 만큼 날짜를 조정
train_time = df_portfolio_train["time"].unique()
train_time = train_time[timewin -1 :]


# 예시: UBAH와 EIIE의 테스트 결과가 리스트 형태라고 가정합니다.
ubah_values = UBAH_results['train']['value']
eiiE_values = EIIE_results['train']['value']

# 데이터프레임의 "time" 컬럼을 인덱스로 사용하여 Series 생성
ubah_price = pd.Series(ubah_values, index=train_time)
eiiE_price = pd.Series(eiiE_values, index=train_time)

# 로그 차분을 사용해 로그 수익률 계산
ubah_return = np.log(ubah_price).diff().dropna()
eiiE_return = np.log(eiiE_price).diff().dropna()

# Quantstats를 사용하여 Sharpe Ratio 계산
sharpe_ubah = qt.stats.sharpe(ubah_return)
sharpe_eiiE = qt.stats.sharpe(eiiE_return)

print("UBAH Sharpe Ratio:", sharpe_ubah)
print("EIIE Sharpe Ratio:", sharpe_eiiE)






# 중복 제거 후 날짜를 가져오고, timewin + 1 만큼 날짜를 조정
test_time = df_portfolio_test["time"].unique()
test_time = test_time[timewin -1 :]


# 예시: UBAH와 EIIE의 테스트 결과가 리스트 형태라고 가정합니다.
ubah_testvalues = UBAH_results['test']['value']
eiiE_tsetvalues = EIIE_results['test']['value']

# 데이터프레임의 "time" 컬럼을 인덱스로 사용하여 Series 생성
ubah_test_price = pd.Series(ubah_testvalues, index=test_time)
eiiE_test_price = pd.Series(eiiE_tsetvalues, index=test_time)

# 로그 차분을 사용해 로그 수익률 계산
ubah_test_return = np.log(ubah_test_price).diff().dropna()
eiiE_test_return = np.log(eiiE_test_price).diff().dropna()

# Quantstats를 사용하여 Sharpe Ratio 계산
sharpe_ubah_test = qt.stats.sharpe(ubah_test_return)
sharpe_eiiE_test = qt.stats.sharpe(eiiE_test_return)

print("UBAH Sharpe Ratio test:", sharpe_ubah_test)
print("EIIE Sharpe Ratio test:", sharpe_eiiE_test)



test_buy_hold_dfw = pd.DataFrame({"value": UBAH_results["test"]["weights"]}, index=test_time)
test_eiie_dfw = pd.DataFrame({"value": EIIE_results["test"]["weights"]}, index=test_time)


import pandas as pd

weights_list = EIIE_results["train"]["weights"]

# 각 배열을 DataFrame으로 변환한 후, axis=1로 결합하면 (n, m) 형태의 DataFrame이 됩니다.
df = pd.concat([pd.DataFrame(arr) for arr in weights_list], axis=1)
