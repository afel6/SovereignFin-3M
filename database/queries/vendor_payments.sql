-- params: ticker, start_date, end_date, limit
SELECT v.vendor_id,
       v.vendor_name,
       v.category,
       v.country,
       COUNT(*)                                         AS txn_count,
       SUM(t.amount)                                    AS total_amount,
       ROUND(AVG(t.paid_on - t.invoice_date), 2)        AS avg_days_to_pay,
       MAX(t.paid_on)                                   AS last_paid_on
FROM transactions t
JOIN companies c ON c.company_id = t.company_id
JOIN vendors v ON v.vendor_id = t.vendor_id
WHERE c.ticker = %(ticker)s::text
  AND (%(start_date)s::date IS NULL OR t.invoice_date >= %(start_date)s::date)
  AND (%(end_date)s::date IS NULL OR t.invoice_date <= %(end_date)s::date)
GROUP BY v.vendor_id
ORDER BY total_amount DESC, v.vendor_id
LIMIT %(limit)s::int
