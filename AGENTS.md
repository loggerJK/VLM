# Repository Agent Instructions

## Work Log

- 작업이 끝날 때마다 저장소 루트의 `CONTEXT.md`를 업데이트한다.
- 새 작업 내역은 `CONTEXT.md`의 기존 항목을 삭제하지 말고 위쪽에 누적한다.
- 각 항목에는 가능한 한 날짜, 목표, 변경 파일, 실행 방법, 검증 결과, 커밋/푸시 정보를 적는다.
- 작업 중 발견한 환경 이슈나 후속 실행 주의사항도 `CONTEXT.md`에 남긴다.

## Current Context

- 최근 작업: Qwen fine-tune 파이프라인에 `rel_position` HF task 추가.
- 학습 실행 위치: `/mnt/data1/jiwon/Qwen3-VL/qwen-vl-finetune`
- 학습 스크립트: `scripts/sft_7b_rel_position.sh`
