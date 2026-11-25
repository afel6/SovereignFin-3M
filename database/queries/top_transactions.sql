-- params: ticker, start_date, end_date, limit
SELECT t.txn_id,
       c.ticker,
       t.invoice_number,
       t.invoice_date,
       t.due_date,
       t.paid_on,
       t.amount,
       v.vendor_id,
       v.vendor_name,
       t.payment_method
FROM transactions t
JOIN companies c ON c.company_id = t.company_id
JOIN vendors v ON v.vendor_id = t.vendor_id
WHERE (%(ticker)s::text IS NULL OR c.ticker = %(ticker)s::text)
  AND (%(start_date)s::date IS NULL OR t.invoice_date >= %(start_date)s::date)
  AND (%(end_date)s::date IS NULL OR t.invoice_date <= %(end_date)s::date)
ORDER BY t.amount DESC, t.txn_id
LIMIT %(limit)s::int
