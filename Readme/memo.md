거의 완전 자동화
완전 자동화도 그냥 스크립트 살짝 수정하면 가능함


일단 제가 여러모로 알아본 바로 gemma4 31b 는 완전 무료입니다.
그리고 api key 차원에서도 이미 비용 한도 1000원으로 걸어놔서 갑자기 요금 폭탄 맞을 일도 없을 것 같네요. 팀 계정도 요금 제한 일단 걸고 쓰겠습니다.
무료인데 이걸 together ai같은 프로바이더들이 왜 유료로 제공해주느냐?... 에 대한 제 답은, 기본 제공 사용량보다 더 헤비하게 쓰고 싶은 사람들이 돈 내고 쓰는게 아닐까요? 무료로는 한도가 있긴 해서요. 잘 모르겠습니다.

데이터셋 생성 속도는 현실적으로 봤을 때, 그냥 계속 돌린다면 하루 600 에서 1000개 정도입니다. 
batch 기능도 있다는데 현재 설계하곤 맞지 않습니다.
무료라도 rpd, rph, tpr 등등 제한은 있지만, 데이터셋 생성 속도 생각해보면 제한의 10%도 채 못 씁니다.
더 빨리 생성해야된다면 제미나이 등 유료 모델을 쓰는 방법도 있겠죠, 다만 챗지피티로 마스터 샘플 200개 정도 뽑았는데, 샘플 개당 40초 정도 걸리고 에러율이 약 5%는 됩니다.
현재 gemma4는 정성적으로 봐도 출력이 너무 만족스럽고, 
validator가 상당히 빡빡한데도 리젝 먹은 적이 한번도 없습니다(샘플 약 30~40개 정도 생성, 프롬프트 수정이 필요해서 지워진 것도 많음)


스킬 쪽 명령이라면?
특히 스킬 쪽 상황을 특정해서 데이터셋 만들고 싶다면
스킬 형식 오버로드 하는게 좋음.
지금 데이터베이스 관점에서 설계가 살짝 잘못됐는데 크리티컬하진 않아서 그냥 들고 감. 거의 이슈가 아닌 수준



py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0001_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0002_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0003_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0004_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0005_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0006_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0007_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0008_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0009_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0010_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0011_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0012_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0013_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0014_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0015_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0016_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0017_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0018_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0019_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0020_raw.jsonl --refresh-report
py -3.11 scripts/sft_cli.py validate --input raw_generations/seed_0021_raw.jsonl --refresh-report







