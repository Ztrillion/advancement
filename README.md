# 3D LiDAR Data advancement Project
Point Cloud Data advancement Project

- 학습데이터셋 PCD, JSON 각 315,252EA
- 클래스 통합 후 CLASS NUM = 7
- Framework = torch 1.12.1 + cu113
  - requirements.txt 만듬, python3.8 가상환경 세팅 후 pip install -r requirements.txt로 환경 세팅 완료 
  
2022/10/12
 ![image](https://user-images.githubusercontent.com/85321962/209616767-77271046-403c-4413-8aa6-b038b5a55b7c.png)

2022/11/23
 ![image](https://user-images.githubusercontent.com/85321962/209616865-f2618643-3814-4b6f-aef9-8d4fc9d295ee.png)



- Class 통합 -> 데이터 편향성 비율 
- PCD & JSON Cloud Point 매핑 에러 -> 데이터 재가공 및 재정제 
- mIoU 90% 초과로 목표치 과달성

2022/12/09
 ![image](https://user-images.githubusercontent.com/85321962/209616914-ab70f7b6-477a-4190-b91e-505f089d0eb6.png)
