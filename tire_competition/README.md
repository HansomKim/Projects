# Hankook Tire 제조 결함 분류 (Tire Defect Prediction)

KAIST–UNIST–POSTECH–Hankook Tire 공동 주최 데이터사이언스 해커톤 프로젝트입니다. (진행 중)
타이어 제조 공정 센서 데이터를 기반으로 양품(Good)/불량(NG)을 예측하는 이진 분류 모델을 개발했습니다.

## 접근 방법
- **모델**: LightGBM (`LGBMClassifier`, n_estimators=500, learning_rate=0.03, num_leaves=63)
- **특성 공학**: 공정(Proc_) / 좌표(X, Y) / 게이지(G) 계열 컬럼별로 평균·표준편차 집계 피처 추가
- **범주형 인코딩**: train/test 공통 vocabulary 기준 category → code 매핑

## 핵심 포인트 — 비용 기반 임계값 최적화
단순 정확도가 아니라 **오분류 비용이 비대칭적인 제조 현장 상황**을 반영해 의사결정 임계값을 직접 설계했습니다.

```python
def compute_profit(pred, true):
    good_correct = ((pred==0) & (true==0)).sum()
    ng_correct   = ((pred==1) & (true==1)).sum()
    false_block  = ((pred==1) & (true==0)).sum()
    return good_correct*100 + ng_correct*20000 - false_block*20000
```

불량품을 놓치는 비용과 양품을 불량으로 오판해 폐기하는 비용이 크게 다르다는 점에 착안해,
0.01~0.30 구간을 탐색하며 기대 이익을 최대화하는 threshold를 선정했습니다.

## 파일
- [`tire_trial_4.ipynb`](./tire_trial_4.ipynb) — 전체 학습·평가·threshold 최적화 코드

## 상태
🔄 진행 중인 대회입니다. (2개 팀 참가)
