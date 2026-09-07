### agent-retrieval (retrieval only, no generation, k=10)

| slice | n | exact | router | ret n | hit@1 | hit@5 | hit@10 | ndcg@1 | ndcg@5 | ndcg@10 | abstain | cite ok | cite gold | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| numeric | 80 | -- | -- | 80 | 0.0% | 0.0% | 0.0% | 0.000 | 0.000 | 0.000 | -- | -- | -- | 0 |
| narrative | 60 | -- | -- | 60 | 16.7% | 45.0% | 60.0% | 0.167 | 0.301 | 0.361 | -- | -- | -- | 0 |
| unanswerable | 10 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| overall | 150 | -- | -- | 140 | 7.1% | 19.3% | 25.7% | 0.071 | 0.129 | 0.155 | -- | -- | -- | 0 |
