## 2026-01-05 (Final Synchronization) - COMPLETE
*(Task performed by **cvlab12 agent**)*

### 1. 프롬프트 정합성 이슈 최종 해결 (Alignment Verified)
- **조치**: 사용자 제공 마스터 메타데이터(`/mnt/data1/jiwon/evaluation_metadata_count.jsonl`)를 기준으로 모든 시각화 및 평가 프로세스 재정렬.
- **결과**: 인퍼런스 결과 폴더(`00000-00089`)와 메타데이터 프롬프트 간의 1:1 대응 완벽 일치 확인.

### 2. 정성적 비교 시각화 (Qualitative Comparison)
- **그리드 생성**: 90개 프롬프트 전용 비교 그리드(4 rows x 6 models) 재생성 완료.
- **라벨링**: 모델명 + 데이터셋 종류 + Training Step 명시 (`Baseline`, `Full FT`, `Trans FT`, `Trans Only (Count/Concat/Point)`).
- **가독성**: 고해상도 출력을 위해 폰트 크기 확대 (Header 40px, Label 30px).
- **결과 위치**: `/mnt/data1/jiwon/Janus/qual_comparison_results/`

### 3. 최종 정량 평가 및 요약 (Final Metrics)
- **평가 환경**: GPU 0번 (`PCI_BUS_ID` 순서 보정)을 사용하여 고속 평가 완료.
- **요약 CSV**: `/mnt/data1/jiwon/geneval/results/janus/summary_metrics.csv` (최종 점수 반영).
- **Confusion Matrix**: 각 모델별 Heatmap 이미지 생성 완료.



---

## 작업 히스토리 (Previous History)

### 2026-01-05 (Initial) - cvlab12 agent
- Metadata generation (2-4 counts) and inference on initial checkpoints.

### 2026-01-05 (Mid) - cvlab22 agent
- Extension to 2-10 counts.
- Baseline inference and metadata verification.
- Found metadata order inconsistency and decided to reset.

---

## 현재 표준 (Current Standards for All Models)

### 1. Master Metadata
- **Path**: `/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl`
- **Classes (10)**: `person, dog, cat, car, cup, chair, book, bottle, apple, tie`
- **Order**: **COUNT-FIRST** (e.g., all class 2s first, then all class 3s...)
- **Total Prompts**: 90

### 2. Result Structure (Consistent across 6 models)
- **Directory**: `/mnt/data1/jiwon/Janus/results_geneval_count_extended/{model_name}/`
- **Index Range**: `00000` to `00089` (Matches Master Metadata line by line)
- **Verification**: Script checks for exactly 90 folders per model.

### 3. Target Models (6)
1. `janus_pro_7b_baseline`
2. `train_full_ckpt600`
3. `train_transformer_only_count_ckpt12000`
4. `train_transformer_only_point_count_ckpt15000`
5. `train_transformer_only_points_ckpt15000`
6. `train_transformer_ckpt600`