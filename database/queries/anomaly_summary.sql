-- params: ticker, min_score
SELECT a.model_version,
       COUNT(*)                                          AS scored_count,
       COUNT(*) FILTER (WHERE a.score >= %(min_score)s::real) AS flagged_count,
       MAX(a.score)                                      AS max_score,
       AVG(a.score)                                      AS avg_score,
       MAX(a.data_snapshot)                              AS data_snapshot,
       MAX(a.scored_at)                                  AS last_scored_at
FROM anomaly_scores a
JOIN transactions t ON t.txn_id = a.txn_id
JOIN companies c ON c.company_id = t.company_id
WHERE %(ticker)s::text IS NULL OR c.ticker = %(ticker)s::text
GROUP BY a.model_version
ORDER BY a.model_version
