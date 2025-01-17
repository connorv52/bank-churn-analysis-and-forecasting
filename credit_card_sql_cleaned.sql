-------------------
-- To clean the data (if even needed) we first need to create a table capable of 
-- importing all of our data in the csv file. 

CREATE TABLE public.credit_card_data (
    CLIENTNUM INTEGER,                      -- Unique client identifier
    Attrition_Flag TEXT,                   -- Customer status (e.g., "Existing Customer", "Attrited Customer")
    Customer_Age INTEGER,                  -- Age of the customer (renamed from Customer_age)
    Gender TEXT,                           -- Gender (e.g., "M", "F")
    Dependent_count INTEGER,               -- Number of dependents
    Education_Level TEXT,                  -- Education level (e.g., "Graduate", "High School")
    Marital_Status TEXT,                   -- Marital status (e.g., "Married", "Single")
    Income_Category TEXT,                  -- Income category (e.g., "$40K-$60K", "Less than $40K")
    Card_Category TEXT,                    -- Type of card (e.g., "Blue", "Gold")
    Months_on_book INTEGER,                -- Period of relationship w/ bank
    Total_Relationship_Count INTEGER,      -- # of products held by customer
    Months_Inactive_12_mon INTEGER,        -- # months inactive in last 12 months
    Contacts_Count_12_mon INTEGER,         -- # of contacts in last 12 months
    Credit_Limit NUMERIC,                  -- Credit limit for the account
    Total_Revolving_Bal INTEGER,           -- Total revolving balance
    Avg_Open_To_Buy NUMERIC,               -- Average open-to-buy credit
    Total_Amt_Chng_Q4_Q1 NUMERIC,          -- Change in transaction amount from Q4 to Q1
    Total_Trans_Amt INTEGER,               -- Total transaction amount
    Total_Trans_Ct INTEGER,                -- Total transaction count
    Total_Ct_Chng_Q4_Q1 NUMERIC,           -- Change in transaction count (Q4 over Q1) (renamed from Total_Ct_Chnq_Q4_Q1)
    Avg_Utilization_Ratio NUMERIC          -- Average card utilization ratio
);

SELECT * FROM credit_card_data
LIMIT 10;

-- Quick look at the data, seems to have been imported correctly. So,
-- we can proceed with cleaning the data. 

-- 1. Remove duplicates
-- 2. Standardize the data (spelling, whitespace, etc.)
-- 3. Null values (blank values)
-- 4. Remove any unnecessary or redundant columns

-- Create a staging table to clean the data from
CREATE TABLE credit_card_data_staging AS
SELECT * FROM credit_card_data;

--1. 
SELECT CLIENTNUM,
	COUNT(*) AS duplicate_count
FROM credit_card_data_staging
GROUP BY CLIENTNUM
HAVING COUNT(*) > 1;

-- There appear to be no duplicates, but we'll verify that
-- using another way to extract potential duplicates

WITH duplicate_cte AS (
SELECT * ,
ROW_NUMBER() OVER(
PARTITION BY CLIENTNUM) AS row_num
FROM credit_card_data_staging
)
SELECT * FROM duplicate_cte
WHERE row_num > 1;

-- There are no rows with a CLIENTNUM ID exceeding 1, 
-- so we can confirm that there are no duplicates in the dataset

-- 2. 
SELECT * FROM credit_card_data
LIMIT 10;

-- Just looking at a snapshot of the data we can see some 
-- trailing whitespace, so we'll fix that on any text columns
-- that we have

-- Verify that we aren't removing necessary spacing in some
-- categories
SELECT income_category, TRIM(income_category)
FROM credit_card_data_staging;


UPDATE credit_card_data_staging
SET Attrition_Flag = TRIM(Attrition_Flag),
	Gender = TRIM(Gender),
    Education_Level = TRIM(Education_Level),
    Marital_Status = TRIM(Marital_Status),
    Income_Category = TRIM(Income_Category),
    Card_Category = TRIM(Card_Category);

SELECT * FROM credit_card_data_staging
LIMIT 10;

-- Already looks much better! Let's continue 

UPDATE credit_card_data_staging
SET Gender = 'Male'
WHERE Gender IN ('M', 'male', 'MALE');

UPDATE credit_card_data_staging
SET Gender = 'Female'
WHERE Gender IN ('F', 'female', 'FEMALE');

UPDATE credit_card_data_staging
SET Marital_Status = 'Married'
WHERE Marital_Status IN ('married', 'MARRIED', 'Married');

UPDATE credit_card_data_staging
SET Marital_Status = 'Single'
WHERE Marital_Status IN ('single', 'SINGLE', 'Single');

SELECT * FROM credit_card_data_staging
LIMIT 10;


-- Giving text columns an initial capitalized letter
UPDATE credit_card_data_staging
SET Attrition_Flag = INITCAP(Attrition_Flag);

UPDATE credit_card_data_staging
SET Education_Level = INITCAP(Education_Level);

UPDATE credit_card_data_staging
SET Income_Category = INITCAP(Income_Category);

UPDATE credit_card_data_staging
SET Card_Category = INITCAP(Card_Category);

SELECT * FROM credit_card_data_staging
LIMIT 10;

-- Let's make sure all columns have no spelling errors
SELECT DISTINCT Attrition_Flag
FROM credit_card_data_staging;

SELECT DISTINCT Education_Level
FROM credit_card_data_staging;

SELECT DISTINCT Income_Category
FROM credit_card_data_staging;

SELECT DISTINCT Card_Category
FROM credit_card_data_staging;

SELECT DISTINCT Marital_Status
FROM credit_card_data_staging;

SELECT DISTINCT Gender
FROM credit_card_data_staging;

-- Now we can move on to numerical standardization

-- For good measure, we will transform some our financial data columns
-- to numeric to consider decimal places for accuracy and consistency


ALTER TABLE credit_card_data_staging
ALTER COLUMN Total_Revolving_Bal TYPE NUMERIC USING Total_Revolving_Bal::NUMERIC;

ALTER TABLE credit_card_data_staging
ALTER COLUMN Total_Trans_Amt TYPE NUMERIC USING Total_Trans_Amt::NUMERIC;

ALTER TABLE credit_card_data_staging
ALTER COLUMN Total_Trans_Ct TYPE NUMERIC USING Total_Trans_Ct::NUMERIC;

-- Now rounding all numeric columns to two decimal places
UPDATE credit_card_data_staging
SET Credit_Limit = ROUND(Credit_Limit, 2),
	Total_Revolving_Bal = ROUND(Total_Revolving_Bal, 2),
	Avg_Open_To_Buy = ROUND(Avg_Open_To_Buy, 2),
	Total_Amt_Chng_Q4_Q1 = ROUND(Total_Amt_Chng_Q4_Q1, 2),
	Total_Trans_Amt = ROUND(Total_Trans_Amt, 2),
	Total_Trans_Ct = ROUND(Total_Trans_Ct, 2),
	Total_Ct_Chng_Q4_Q1 = ROUND(Total_Ct_Chng_Q4_Q1, 2),
	Avg_Utilization_Ratio = ROUND(Avg_Utilization_Ratio, 2);

SELECT * FROM credit_card_data_staging
LIMIT 10;

-- 3.
-- Since our primary concern here is the churner status, we will 
-- check for any nulls in that particular column first

SELECT COUNT(*) AS null_attrition_flag_count
FROM credit_card_data_staging
WHERE Attrition_Flag IS NULL;

-- No nulls 

-- We will check for nulls elsewhere

SELECT 
    COUNT(*) AS total_rows,
    COUNT(CASE WHEN Gender IS NULL THEN 1 END) AS null_gender,
    COUNT(CASE WHEN Education_Level IS NULL THEN 1 END) AS null_education_level,
    COUNT(CASE WHEN Marital_Status IS NULL THEN 1 END) AS null_marital_status,
    COUNT(CASE WHEN Income_Category IS NULL THEN 1 END) AS null_income_category,
    COUNT(CASE WHEN Card_Category IS NULL THEN 1 END) AS null_card_category,
    COUNT(CASE WHEN Credit_Limit IS NULL THEN 1 END) AS null_credit_limit,
    COUNT(CASE WHEN Total_Revolving_Bal IS NULL THEN 1 END) AS null_total_revolving_bal,
    COUNT(CASE WHEN Avg_Open_To_Buy IS NULL THEN 1 END) AS null_avg_open_to_buy,
    COUNT(CASE WHEN Total_Amt_Chng_Q4_Q1 IS NULL THEN 1 END) AS null_total_amt_chng_q4_q1,
    COUNT(CASE WHEN Total_Trans_Amt IS NULL THEN 1 END) AS null_total_trans_amt,
    COUNT(CASE WHEN Total_Trans_Ct IS NULL THEN 1 END) AS null_total_trans_ct,
    COUNT(CASE WHEN Total_Ct_Chng_Q4_Q1 IS NULL THEN 1 END) AS null_total_ct_chng_q4_q1,
    COUNT(CASE WHEN Avg_Utilization_Ratio IS NULL THEN 1 END) AS null_avg_utilization_ratio
FROM credit_card_data_staging;

-- No nulls across the board. Rare, but does happen. Always need
-- to check regardless in order to avoid confusion when conducting
-- descriptive and predictive analyses later. 

-- 4. We already removed the naive bayes columns in Excel prior 
-- to importing the csv file, so we shouldn't need to drop any columns
-- in this case! However, if we were to have a running total column
-- or any other column in the effort of helping us keep track of potential errors or 
-- inconsistencies, now would be a great time before we move on to the analysis:

-- ALTER TABLE credit_card_data_staging 
-- DROP COLUMN xyz_column;

SELECT * FROM credit_card_data_staging;
-- Data has been cleaned, and we can now safely move on to analyzing the dataset.
-------------------