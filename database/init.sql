-- Applied by `python -m database.rebuild` as a superuser inside the target database, in one
-- transaction, after the roles sf_owner and sf_reader exist. Role passwords are set by the
-- rebuild from the environment, never here.

DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public AUTHORIZATION sf_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO sf_reader;

ALTER ROLE sf_reader SET default_transaction_read_only = on;
ALTER ROLE sf_reader SET statement_timeout = '5s';

SET ROLE sf_owner;

CREATE TABLE companies (
    company_id    smallint PRIMARY KEY,
    ticker        text NOT NULL UNIQUE,
    company_name  text NOT NULL,
    sector        text NOT NULL,
    size_tier     text NOT NULL CHECK (size_tier IN ('S', 'M', 'L'))
);

CREATE TABLE vendors (
    vendor_id           integer PRIMARY KEY,
    vendor_name         text NOT NULL UNIQUE,
    country             text NOT NULL,
    category            text NOT NULL CHECK (category IN (
        'software', 'hardware', 'professional_services', 'logistics',
        'facilities', 'marketing', 'raw_materials', 'travel')),
    onboarded_on        date NOT NULL,
    payment_terms_days  smallint NOT NULL CHECK (payment_terms_days IN (30, 45, 60, 90))
);

CREATE TABLE transactions (
    txn_id          bigint PRIMARY KEY,
    company_id      smallint NOT NULL REFERENCES companies,
    vendor_id       integer NOT NULL REFERENCES vendors,
    invoice_number  text NOT NULL,
    invoice_date    date NOT NULL,
    due_date        date NOT NULL,
    paid_on         date,
    amount          numeric(14, 2) NOT NULL CHECK (amount > 0),
    payment_method  text NOT NULL CHECK (payment_method IN ('ach', 'wire', 'check', 'card')),
    UNIQUE (vendor_id, invoice_number),
    CHECK (due_date >= invoice_date)
);

CREATE TABLE anomaly_scores (
    txn_id         bigint      REFERENCES transactions,
    model_version  text,
    score          real        NOT NULL CHECK (score BETWEEN 0 AND 1),
    reasons        text[]      NOT NULL,
    data_snapshot  bigint      NOT NULL,  -- generator seed-run epoch or a monotonically increasing build number
    scored_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (txn_id, model_version)
);

CREATE INDEX transactions_company_invoice_idx ON transactions (company_id, invoice_date);
CREATE INDEX transactions_vendor_idx ON transactions (vendor_id);

GRANT SELECT ON ALL TABLES IN SCHEMA public TO sf_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE sf_owner IN SCHEMA public GRANT SELECT ON TABLES TO sf_reader;

RESET ROLE;
