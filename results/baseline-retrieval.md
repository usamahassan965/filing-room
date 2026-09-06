### baseline-retrieval (retrieval only, no generation, k=10)

| slice | n | exact | router | hit@1 | hit@5 | hit@10 | ndcg@1 | ndcg@5 | ndcg@10 | abstain | cite ok | cite gold | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| numeric | 80 | -- | -- | 2.5% | 2.5% | 5.0% | 0.025 | 0.008 | 0.011 | -- | -- | -- | 0 |
| narrative | 60 | -- | -- | 10.0% | 15.0% | 30.0% | 0.100 | 0.105 | 0.157 | -- | -- | -- | 0 |
| unanswerable | 10 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| overall | 150 | -- | -- | 5.7% | 7.9% | 15.7% | 0.057 | 0.050 | 0.074 | -- | -- | -- | 0 |
