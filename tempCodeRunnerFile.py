

"""
Overview Insights :-

**shape =>

-- Rows = 614 , Columns = 13 -- 12 After removing a column Loan_ID

**Empty Values =>

-- 149 Null values

**Duplicated values =>
-- Zero value

**Dtypes =>
-- Object(8) , floats64 (4) , int64(1)

Outliers =>
-- there are 4 columns have outliers

## Actions

-- Deleted not useful column 'Loan_ID'
-- Optimized Data with category , float32 , int8
-- Filled Null values with categorical columns--> Mode , Numerical Columns --> Median
-- Mapping the target column Yes : 1 , No : 0
-- Most Repeated Yes and No Loan status Yes = 422 , No = 192

"""

