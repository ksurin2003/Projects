# FDA KOSPI200 Daily Student Dataset

## 1. 데이터셋 목적

본 데이터셋은 금융데이터분석(Financial Data Analysis) 수업에서
한국 주식시장의 일별 가격, 수익률, 거래량 및 시가총액을 이용한
EDA, 기술적 분석, 수익률 분석 및 백테스트 실습을 수행하기 위한
학생용 데이터셋이다.

분석 대상 universe는 KOSPI200이다.

---

## 2. 파일 구성

### FDA_kospi200_daily.parquet

실제 분석에 사용하는 핵심 일별 패널 데이터이다.

### FDA_kospi200_snapshot.parquet

기업명, 시장, 산업, 상장일 등 비교적 정적인 기업 metadata를
제공한다.

이 파일의 정보는 현재 또는 특정 기준일의 snapshot 정보이며,
역사적 point-in-time 기업 특성을 의미하지 않는다.

### FDA_kospi200_daily_data_dictionary.csv

daily parquet에 포함된 각 변수의 정의, 단위, dtype 및
주의사항을 설명한다.

### FDA_kospi200_daily_sample.csv

전체 parquet의 구조를 Excel 등에서 쉽게 확인하기 위한
소규모 예제 파일이다.

---

## 3. Observation Grain

본 데이터셋에서 한 행은

$$
(\text{Date}, \text{Symbol})
$$

로 정의된다.

즉,

> 한 행 = 특정 거래일의 특정 종목에 대한 관측치

이다.

(Date, Symbol)은 데이터셋 내에서 유일해야 한다.

---

## 4. 표본기간

관측 데이터 기준:

- 시작일: 2001-01-02
- 종료일: 2025-12-30
- 전체 고유 종목 수: 527
- Snapshot 종목 수: 527

공식 학생용 universe contract는
2001-01부터 2025-12까지의 KOSPI200 membership을 사용한다.

---

## 5. KOSPI200 Universe 정의

본 데이터셋은 현재 시점의 KOSPI200 구성종목을 과거에
소급하여 사용하지 않는다.

각 월의 실제 KOSPI200 membership을 사용하는
monthly point-in-time universe이다.

따라서 미래에 KOSPI200에 편입될 종목을 과거 시점부터
KOSPI200 구성종목으로 간주하는 survivorship bias를 방지한다.

`MembershipMonth`는 각 관측치에 적용된 KOSPI200
membership 기준월을 나타낸다.

`KOSPI200_Member`는 본 학생용 파일에서 항상 1이다.

---

## 6. 가격 변수 사용 원칙

### Close

당시 실제 시장에서 관측된 종가를 확인할 때 사용한다.

### AdjustedClose

주식분할, 병합 등 기업행동의 영향을 조정한 가격계열이다.

기간 간 가격 변화나 기술적 지표를 계산할 때 기본적으로
AdjustedClose를 사용한다.

예:

- 이동평균
- Momentum
- RSI
- MACD
- Bollinger Band
- 기간 수익률 계산

---

## 7. 수익률 변수

### AdjustedReturnPct

수정주가를 기준으로 계산된 일별 수익률이다.

데이터의 단위는 percent이다.

Python에서 decimal return으로 사용할 경우

$$
r_t
=
\frac{\text{AdjustedReturnPct}}{100}
$$

으로 변환한다.

### AdjustedReturn_TotalReturn_Pct

주가 변화뿐 아니라 현금배당까지 포함한 총수익률
(total return)이다.

개념적으로

$$
R_t^{TR}
=
\frac{P_t + D_t}{P_{t-1}} - 1
$$

과 같은 투자성과를 측정한다.

따라서 장기 투자성과 또는 포트폴리오 성과를 평가할 때
AdjustedReturn_TotalReturn_Pct를 사용하는 것이 적절하다.

---

## 8. Parquet 불러오기

```python
import pandas as pd

df = pd.read_parquet(
    "FDA_kospi200_daily.parquet"
)

print(df.head())
print(df.shape)
print(df.dtypes)
```

---

## 9. 특정 종목 선택

```python
samsung = df.loc[
    df["Symbol"] == "005930"
].copy()

samsung = samsung.sort_values("Date")
```

종목코드는 숫자가 아니라 6자리 문자열로 처리한다.

---

## 10. 수정주가를 이용한 단순수익률 계산

```python
samsung["Return"] = (
    samsung["AdjustedClose"]
    .pct_change()
)
```

데이터에 이미 저장된 수정수익률을 decimal 형태로 사용할 경우:

```python
samsung["Return"] = (
    samsung["AdjustedReturnPct"] / 100
)
```

---

## 11. 결측치 주의사항

시장 데이터에는 다음과 같은 이유로 결측치가 존재할 수 있다.

- 거래정지
- 신규상장
- 상장폐지
- 기업행동
- 데이터 공급원 특성

결측치를 임의로 0으로 바꾸지 않는다.

특히 수익률 NaN은 0% 수익률과 의미가 다르다.

---

## 12. Snapshot 주의사항

`FDA_kospi200_snapshot.parquet`에 포함된 기업 metadata는
현재 또는 특정 기준일의 snapshot이다.

따라서 해당 파일의 현재 산업분류나 현재 KOSPI200 flag를
과거 날짜의 point-in-time 정보로 해석해서는 안 된다.

역사적 KOSPI200 여부는 반드시 daily dataset의
`MembershipMonth` 기준 universe를 사용한다.

---

## 13. 핵심 원칙

1. 한 행은 Date × Symbol이다.
2. Symbol은 6자리 문자열이다.
3. KOSPI200 universe는 point-in-time이다.
4. 현재 구성종목을 과거에 소급하지 않는다.
5. 기술적 분석에서는 기본적으로 AdjustedClose를 사용한다.
6. 수익률 단위가 percent인지 decimal인지 반드시 확인한다.
7. 배당 포함 장기 투자성과에는 Total Return을 사용한다.
8. 결측치를 자동으로 0으로 대체하지 않는다.
