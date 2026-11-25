-- params: ticker
SELECT c.ticker,
       c.company_name,
       c.sector,
       c.size_tier,
       COUNT(DISTINCT t.vendor_id)                      AS vendor_count,
       COUNT(t.txn_id)                                  AS txn_count,
       COALESCE(SUM(t.amount), 0)                       AS total_amount,
       COUNT(t.txn_id) FILTER (WHERE t.paid_on IS NULL) AS unpaid_count,
       MIN(t.invoice_date)                              AS first_invoice_date,
       MAX(t.invoice_date)                              AS last_invoice_date
FROM companies c
LEFT JOIN transactions t ON t.company_id = c.company_id
WHERE c.ticker = %(ticker)s::text
GROUP BY c.company_id
ORDER BY c.company_id
