### agent-retrieval-norerank (retrieval only, no generation, k=10)

| slice | n | exact | router | ret n | hit@1 | hit@5 | hit@10 | ndcg@1 | ndcg@5 | ndcg@10 | abstain | cite ok | cite gold | LLM calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| numeric | 80 | -- | -- | 80 | 1.2% | 5.0% | 7.5% | 0.013 | 0.012 | 0.015 | -- | -- | -- | 0 |
| narrative | 60 | -- | -- | 60 | 20.0% | 51.7% | 63.3% | 0.200 | 0.349 | 0.390 | -- | -- | -- | 0 |
| unanswerable | 10 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0 |
| overall | 150 | -- | -- | 140 | 9.3% | 25.0% | 31.4% | 0.093 | 0.157 | 0.175 | -- | -- | -- | 0 |
