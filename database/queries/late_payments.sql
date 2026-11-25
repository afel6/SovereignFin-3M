-- params: ticker, start_date, end_date, min_days_late, limit
SELECT c.ticker,
       v.vendor_id,
       v.vendor_name,
       COUNT(*)                                    AS late_count,
       ROUND(AVG(t.paid_on - t.due_date), 2)       AS avg_days_late,
       MAX(t.paid_on - t.due_date)                 AS max_days_late,
       SUM(t.amount)                               AS late_amount
FROM transactions t
JOIN companies c ON c.company_id = t.company_id
JOIN vendors v ON v.vendor_id = t.vendor_id
WHERE (%(ticker)s::text IS NULL OR c.ticker = %(ticker)s::text)
  AND (%(start_date)s::date IS NULL OR t.invoice_date >= %(start_date)s::date)
  AND (%(end_date)s::date IS NULL OR t.invoice_date <= %(end_date)s::date)
  AND t.paid_on - t.due_date > %(min_days_late)s
GROUP BY c.company_id, v.vendor_id
ORDER BY late_count DESC, c.ticker, v.vendor_id
LIMIT %(limit)s::int
