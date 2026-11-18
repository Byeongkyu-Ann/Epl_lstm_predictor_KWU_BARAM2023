# 2단계 LSTM을 이용한 EPL 경기 결과 예측

[English](README.md) | **한국어**

두 팀 정보만으로 영국 프리미어리그 경기 결과를 예측하는 PyTorch 프로젝트입니다.
광운대학교 로봇 동아리 BARAM에서 2023년 9월부터 11월까지 진행했습니다.

## 개요

경기 전에는 두 팀 정보만 알 수 있습니다. 결과를 설명하는 경기 내 통계는 알 수 없습니다.
이 프로젝트는 두 단계의 LSTM으로 이 간극을 메웁니다.

1. **1단계**는 두 팀으로부터 경기 통계 5종의 홈팀 비율을 예측합니다.
2. **2단계**는 두 팀과 1단계 통계로부터 승, 무, 패를 예측합니다.

```
홈팀, 원정팀
        |
        v
  1단계 (LSTM 5개)  ->  점유율, 슈팅, 볼터치, 패스, 클리어링 (홈팀 비율)
        |
        v
  2단계 (LSTM 1개)  ->  홈팀 승 / 무 / 패
```

데이터는 premierleague.com에서 Selenium으로 수집합니다.

## 저장소 구조

```
eplpredict/
  config.py          경로, 팀 인덱스, 컬럼 이름
  crawl.py           한 시즌 크롤러 (Selenium)
  preprocess.py      원시 경기 데이터 -> 입력 특징
  model.py           LSTM 모델과 체크포인트 함수
  data.py            시간순 분할과 시퀀스 윈도우
  training.py        전체 배치 학습 루프
  train_stage1.py    통계 예측 모델 5개 학습
  train_stage2.py    결과 예측 모델 학습
  predict.py         2단계 추론과 명령행 도구
  app.py             PyQt5 데스크톱 데모
experiments/
  wdl_from_match_stats.py   실제 경기 통계로 결과 예측
assets/              데모용 Qt Designer 파일
data/                CSV 파일. 버전 관리하지 않음
models/              체크포인트. 버전 관리하지 않음
```

## 빠른 시작

```bash
pip install -r requirements.txt
```

**1. 크롤링.** 시즌마다 한 번씩 실행합니다. 시즌 ID는 결과 페이지 URL의 `se` 값입니다.
Chrome이 필요합니다.

```bash
python -m eplpredict.crawl --season-id 489 --out data/raw_2022_23.csv
```

**2. 전처리.** 파일은 오래된 시즌부터 순서대로 나열합니다.

```bash
python -m eplpredict.preprocess --inputs data/raw_2021_22.csv data/raw_2022_23.csv
```

**3. 학습.**

```bash
python -m eplpredict.train_stage1
python -m eplpredict.train_stage2 --plot
```

**4. 예측.**

```bash
python -m eplpredict.predict "Man Utd" "Man City"
python -m eplpredict.app        # 데스크톱 데모. assets/first2.ui 필요
```

## 데이터와 입력 특징

크롤러는 경기당 한 행을 저장합니다. 필드는 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| `home_team`, `away_team` | 팀 이름 |
| `goal_diff` | 홈팀 득점 - 원정팀 득점 |
| `home_`, `away_` + 통계 이름 | 점유율, 슈팅, 유효슈팅, 볼터치, 패스, 태클, 클리어링 |

전처리 단계에서 다음 컬럼을 추가합니다.

| 컬럼 | 정의 |
|---|---|
| `home_idx`, `away_idx` | 1부터 35까지의 팀 인덱스. 값이 클수록 강팀입니다. `config.py` 참고 |
| `possession_share`, `shot_share`, `touch_share`, `pass_share`, `clearance_share` | 홈팀 / (홈팀 + 원정팀) |
| `result` | 0 = 홈팀 패, 1 = 무승부, 2 = 홈팀 승 |

경기 순서는 섞지 않습니다. 앞쪽 90%로 학습하고 뒤쪽 10%로 평가합니다.

## 모델

| | 1단계 | 2단계 |
|---|---|---|
| 개수 | 5 | 1 |
| 입력 | `home_idx`, `away_idx` | 두 팀 인덱스와 1단계 비율 5개 |
| 출력 | 0~1 범위의 홈팀 비율 | 3개 클래스 |
| 손실 함수 | MSE | 교차 엔트로피 |
| LSTM 크기 | 4층, 은닉 유닛 10개 | 6층, 은닉 유닛 30개 |
| 옵티마이저 | Adam, 학습률 0.001 | Adam, 학습률 0.001 |
| 평가 지표 | R² | 정확도 |

2단계는 각 경기의 실제 통계로 학습합니다. 예측할 때는 1단계 출력을 입력으로 받습니다.

## 결과

각 학습 스크립트는 마지막에 평가 지표를 출력합니다. 1단계는 통계별 R²를 출력합니다. 2단계는 정확도와 항상 가장 많은 클래스로 예측했을 때의 정확도를 출력합니다.

`experiments/wdl_from_match_stats.py`는 실제 경기 통계를 알 때의 기준 정확도를 제공합니다.

원래 데모의 출력 예시입니다. 홈팀은 맨체스터 유나이티드이고 원정팀은 맨체스터 시티입니다.

| 통계 | 홈팀 비율 | 원정팀 비율 |
|---|---|---|
| 점유율 | 39.4% | 60.6% |
| 슈팅 | 42.2% | 57.8% |
| 볼터치 | 38.5% | 61.5% |
| 패스 | 35.3% | 64.7% |
| 클리어링 | 55.9% | 44.1% |

예측 결과는 홈팀 패배입니다. 2023년 10월 29일 실제 경기는 0-3으로 끝났습니다.
이 값은 한 경기의 예시이며 성능 지표가 아닙니다.

## 한계

- 모든 샘플의 시퀀스 길이가 1입니다. LSTM은 한 시점만 보며 경기 이력을 사용하지 않습니다.
- 팀은 고정된 인덱스 하나로 표현됩니다. 최근 폼, 라인업, 부상은 사용하지 않습니다.
- 2단계는 실제 통계로 학습하고 실제 통계로 평가합니다. 두 단계를 이은 정확도는 학습 스크립트가 측정하지 않습니다.
- 크롤러는 2023년 가을 사이트 구조의 XPath에 의존합니다. 현재 사이트 구조에서는 테스트하지 않았으며 수정이 필요할 수 있습니다.

## 프로젝트 정보

- 작성자: 안병규, 광운대학교 로봇 동아리 BARAM
- 기간: 2023년 9월 ~ 11월
- 기술 스택: Python, PyTorch, Selenium, pandas, scikit-learn, PyQt5
