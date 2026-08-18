# FREB-CAVER

**[Research] JUDO의 산업 이상 추론을 위한 모델 내부 인과적 시각 증거 결합 어댑터**

FREB-CAVER는 정상 참조 이미지와 질의 이미지의 시각 토큰을 부분 수송으로
정렬하고, 정렬되지 않은 결함 증거와 정상 일치 증거를 분리한 뒤, 그 판단을
JUDO의 `<seg>`, `<think>`, `<answer>` 상태에 저랭크 잔차로 주입합니다. 외부
라우터나 추론 임계값은 사용하지 않으며 JUDO 본체는 동결합니다.

핵심 독립 검증은 자산이 겹치지 않는 MMAD 1,120문항 paired holdout입니다.
전체 정확도는 JUDO 80.0893%에서 FREB-CAVER 81.0714%로 **0.9821%p**
상승했고, exact McNemar `p=0.043285`, cluster bootstrap 95% 구간은
`+0.0908`~`+1.8683%p`였습니다. 다만 AD balanced accuracy는 65.6250%로
변화가 없어, 이상 과잉 탐지 해결에 대한 사전등록 공동 주장은 확인되지
않았습니다.

## MMAD 7개 영역별 JUDO 비교

아래 수치는 모델 선택에 사용되지 않은 자산 비중복 1,120문항 holdout
기준입니다. 각 영역은 GoodsAD, MVTec-AD, MVTec-LOCO, VisA에서 40문항씩
총 160문항으로 구성됩니다.

| MMAD 영역 | JUDO | FREB-CAVER | 변화 |
|---|---:|---:|---:|
| Anomaly Detection | 65.6250% | 65.6250% | 0.0000%p |
| Defect Analysis | 87.5000% | **88.7500%** | **+1.2500%p** |
| Defect Classification | 70.6250% | **71.2500%** | **+0.6250%p** |
| Defect Description | **86.8750%** | 86.2500% | -0.6250%p |
| Defect Localization | 70.0000% | **73.1250%** | **+3.1250%p** |
| Object Analysis | 85.0000% | **87.5000%** | **+2.5000%p** |
| Object Classification | 95.0000% | 95.0000% | 0.0000%p |

7개 영역 중 4개가 향상되고 2개는 동일하며 1개가 하락했습니다. 가장 큰
향상은 Defect Localization `+3.1250%p`, Object Analysis `+2.5000%p`이고,
Defect Description은 `-0.6250%p` 하락했습니다. AD는 변화가 없습니다.

전체 MMAD 39,670문항에서는 28-cell macro 81.9063%, micro 81.2150%를
기록했습니다. 그러나 이 중 6,160문항(15.53%)은 Stage-4 학습 또는 선택에
사용되었으므로 이 수치는 독립 일반화 성능이 아니라 전체 동작 감사 결과로
해석해야 합니다.

설치·실행·세부 실험 설계는 [영문 README](README.md), [방법론](METHOD.md),
[결과](RESULTS.md), [재현 안내](REPRODUCIBILITY.md)를 참고하세요.
