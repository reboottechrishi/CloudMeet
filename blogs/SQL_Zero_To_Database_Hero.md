# SQL Code Reference - From Zero to Database Hero
**Complete Reference Guide by Abhishek Anand**

## Table of Contents
1. [What is Data?](#what-is-data)
2. [Database Fundamentals](#database-fundamentals)
3. [Database Management Systems (DBMS)](#database-management-systems-dbms)
4. [SQL Introduction](#sql-introduction)
5. [Database and Table Operations](#database-and-table-operations)
6. [Data Types and Constraints](#data-types-and-constraints)
7. [Data Manipulation (CRUD Operations)](#data-manipulation-crud-operations)
8. [SQL Clauses](#sql-clauses)
9. [SQL Operators](#sql-operators)
10. [CASE Statements](#case-statements)
11. [Indexes](#indexes)
12. [Aggregate Functions](#aggregate-functions)
13. [JOINS](#joins)
14. [Subqueries](#subqueries)
15. [Window Functions](#window-functions)
16. [Database Design & Normalization](#database-design--normalization)
17. [Views, Stored Procedures & Functions](#views-stored-procedures--functions)
18. [Transactions & ACID Properties](#transactions--acid-properties)
19. [Performance Optimization](#performance-optimization)
20. [Real-world Project Examples](#real-world-project-examples)
21. [Practice Resources](#practice-resources)

---

## What is Data?

**Definition**: Data refers to raw facts, figures, symbols, or statistics collected together for analysis, reference, or processing. It represents the most basic level of information - individual data points that lack context or meaning until they are processed and organized into meaningful patterns.

**Data Characteristics**:
- Raw and unprocessed
- Can exist in many forms: numbers, text, measurements, observations, descriptions, images, audio, and videos
- Has no inherent meaning without context
- Forms the foundation for all information systems
- Can be quantitative (numerical) or qualitative (descriptive)

**Data Hierarchy**: Data → Information → Knowledge → Wisdom

This hierarchy represents the progressive refinement of raw data into actionable insights:

### Detailed Example:
- **Data**: Temperature readings: 25°C, 30°C, 28°C, 26°C, 29°C (raw measurements)
- **Information**: Average temperature this week is 27.6°C, with a range of 5°C (processed data with context)
- **Knowledge**: This temperature range is ideal for plant growth and occurs during optimal growing season (applied understanding)
- **Wisdom**: Schedule watering based on temperature patterns, adjust planting times, and prepare for seasonal changes (actionable decisions)

### Types of Data:

#### 1. Structured Data
- **Definition**: Data organized in a predefined format with clear relationships between data elements
- **Characteristics**: 
  - Fixed schema or format
  - Easily searchable and queryable
  - Fits neatly into rows and columns
  - Machine-readable and processable
- **Storage**: Relational databases (SQL), spreadsheets, data warehouses
- **Examples**: Customer databases, financial records, inventory systems, employee records
- **Advantages**: Easy to analyze, query, and report on
- **Disadvantages**: Rigid structure, difficult to modify schema

#### 2. Unstructured Data
- **Definition**: Data that doesn't follow a predefined format or structure
- **Characteristics**:
  - No fixed schema
  - Human-generated content
  - Rich in context but difficult to process automatically
  - Requires advanced techniques for analysis
- **Storage**: File systems, data lakes, NoSQL databases
- **Examples**: Email content, social media posts, documents, images, videos, audio files, web pages
- **Advantages**: Natural format, rich in context, flexible
- **Disadvantages**: Difficult to query, requires preprocessing, storage intensive

#### 3. Semi-structured Data
- **Definition**: Data that contains organizational properties but doesn't conform to rigid tabular structures
- **Characteristics**:
  - Self-describing structure
  - Uses tags, labels, or markers
  - More flexible than structured, more organized than unstructured
  - Hierarchical or nested organization
- **Storage**: Document databases, XML databases, JSON stores
- **Examples**: JSON files, XML documents, HTML pages, configuration files, log files
- **Advantages**: Balance of structure and flexibility, easier to parse than unstructured data
- **Disadvantages**: Can become complex, varying schemas within the same dataset

---

## Database Fundamentals

### What is a Database?

**Definition**: A database is a structured, organized collection of related data that is stored electronically in a computer system. It serves as a centralized repository where data can be systematically stored, accessed, managed, updated, and analyzed efficiently.

**Key Characteristics of Databases**:
- **Organization**: Data is structured according to a specific model or schema
- **Persistence**: Data survives beyond the execution of programs that created it
- **Concurrency**: Multiple users can access data simultaneously
- **Recovery**: Built-in mechanisms to protect against data loss
- **Security**: Access controls and authentication mechanisms
- **Integrity**: Mechanisms to ensure data accuracy and consistency
- **Scalability**: Ability to handle growing amounts of data and users

**Real-world Database Examples**:
- **Banking Systems**: Store customer accounts, transactions, loan information
- **E-commerce Platforms**: Product catalogs, customer profiles, order history
- **Healthcare Systems**: Patient records, medical history, treatment plans
- **Educational Institutions**: Student records, course information, grades
- **Social Media**: User profiles, posts, connections, interactions
- **Government Systems**: Citizen records, tax information, public services

### Types of Databases:
- Relational Database
- Non-Relational Database
- Distributed Database
- Object-oriented Database
- Hierarchical Database

### Relational vs Non-Relational Databases

| **Relational Database** | **Non-Relational Database** |
|------------------------|------------------------------|
| Stores data in tables (rows & columns) | Uses flexible data models (documents, key-value, graph) |
| Uses SQL for queries | Uses various query languages (MongoDB, Cassandra) |
| Best for structured data and complex relationships | Best for unstructured/semi-structured data and scalability |
| Examples: MySQL, PostgreSQL, Oracle | Examples: MongoDB, Cassandra, Redis |

### Database History
- **1970**: Edgar F. Codd introduced the relational model for databases
- **IBM's System R (1970s)**: One of the first implementations of the relational model
- **SQL (1970s)**: Developed to interact with relational databases, became industry standard

### Disadvantages of File-Based Systems:
- **Data Integrity Issues**: Difficult to maintain without built-in constraints
- **Complex Querying**: Manual searches, no standardized query language
- **Data Redundancy**: Same information stored in multiple places
- **Backup and Recovery**: Slow and cumbersome processes

---

## Database Management Systems (DBMS)

**Definition**: A Database Management System (DBMS) is sophisticated software that serves as an intermediary between users and databases, providing a systematic and organized approach to creating, storing, organizing, retrieving, updating, and managing data. It acts as a comprehensive interface that handles all aspects of database operations while ensuring data integrity, security, and efficient access.

**Core Components of DBMS**:
1. **Database Engine**: Core service for storing, processing, and securing data
2. **Database Schema**: Structure that defines how data is organized
3. **Query Processor**: Interprets and executes database queries
4. **Transaction Manager**: Ensures ACID properties are maintained
5. **Storage Manager**: Manages the storage of data on disk
6. **Security Subsystem**: Handles authentication and authorization

### Key Functions of DBMS:

#### 1. Data Definition (DDL - Data Definition Language)
- **Purpose**: Define the structure and schema of the database
- **Operations**: Create, modify, and delete database objects
- **Components**:
  - **Tables**: Define structure with columns and data types
  - **Relationships**: Establish foreign key relationships between tables
  - **Constraints**: Set rules for data validity (NOT NULL, CHECK, UNIQUE)
  - **Indexes**: Create data structures for faster retrieval
  - **Views**: Define virtual tables based on query results
- **Example**: Creating a table with specific columns, data types, and constraints

#### 2. Data Manipulation (DML - Data Manipulation Language)
- **Purpose**: Perform operations on the actual data stored in databases
- **Core Operations**:
  - **INSERT**: Add new records to tables
  - **SELECT**: Retrieve data based on specified criteria
  - **UPDATE**: Modify existing records
  - **DELETE**: Remove records from tables
- **Advanced Features**:
  - **Joins**: Combine data from multiple tables
  - **Subqueries**: Nested queries for complex data retrieval
  - **Aggregation**: Summarize data using functions like SUM, COUNT, AVG
  - **Filtering**: Use WHERE clauses for conditional operations

#### 3. Security and Access Control
- **Authentication**: Verify user identity through credentials
- **Authorization**: Control what authenticated users can access or modify
- **User Management**: Create and manage user accounts and groups
- **Role-Based Access**: Assign permissions based on user roles
- **Audit Trails**: Log database activities for security monitoring
- **Encryption**: Protect data both at rest and in transit
- **Backup Security**: Secure backup and recovery procedures

#### 4. Data Integrity and Consistency
- **Entity Integrity**: Ensure each record has a unique identifier (Primary Key)
- **Referential Integrity**: Maintain valid relationships between tables (Foreign Keys)
- **Domain Integrity**: Ensure data values fall within acceptable ranges
- **Transaction Integrity**: Maintain data consistency during concurrent operations
- **Constraint Enforcement**: Automatically validate data against defined rules
- **Rollback Mechanisms**: Undo changes that violate integrity rules

**Additional DBMS Functions**:
- **Concurrent Access**: Allow multiple users to access data simultaneously without conflicts
- **Backup and Recovery**: Protect against data loss and restore data when needed
- **Performance Optimization**: Query optimization, indexing strategies, caching
- **Data Independence**: Separate physical storage from logical data structure
- **Metadata Management**: Store information about data structure and relationships

---

## SQL Introduction

SQL (Structured Query Language) is a standard programming language used to manage and manipulate relational databases.

### SQL Categories:

#### Data Definition Language (DDL)
- **CREATE**: Defines a new database object
- **ALTER**: Modifies an existing database object
- **DROP**: Deletes an existing database object

#### Data Manipulation Language (DML)
- **SELECT**: Retrieves data from a database
- **INSERT**: Adds new records to a table
- **UPDATE**: Modifies existing data in a table
- **DELETE**: Removes records from a table

---

## Database and Table Operations

### Create Database
```sql
CREATE DATABASE Projectdb;
```

### Create Table (Basic)
```sql
CREATE TABLE table_name(
    col_1 INT,
    col_2 varchar(100),
    col_3 varchar(50)
);
```

### Drop Table
```sql
DROP TABLE table_name;
```

### Drop Database
```sql
DROP DATABASE Projectdb;
```

---

## Data Types and Constraints

### Common Data Types in SQL:
- **INT**: Stores whole numbers without decimals
- **FLOAT**: Stores decimal values
- **VARCHAR(n)**: Stores variable-length string of maximum n size
- **DATETIME**: Stores date and time combined

### Constraints in SQL:
Constraints are rules enforced on data columns to ensure data integrity and accuracy.

- **NOT NULL**: Ensures a column cannot have empty values
- **PRIMARY KEY**: Uniquely identifies each record
- **FOREIGN KEY**: Links two tables together
- **UNIQUE**: Ensures all values in a column are different
- **CHECK**: Ensures values meet a specific condition
- **DEFAULT**: Sets a default value when no value is specified

### Create Table with Constraints
```sql
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INT CHECK (age >= 18),
    enrollment_date DATE DEFAULT GETDATE()
);
```

### Types of Keys in SQL:

#### Primary Key
**Definition**: A primary key is a column or combination of columns that uniquely identifies each row in a database table. It serves as the main identifier for records and ensures that no two rows can have identical primary key values.

**Characteristics**:
- **Uniqueness**: No duplicate values allowed
- **Non-null**: Cannot contain NULL values
- **Immutability**: Values should not change once assigned
- **Minimal**: Should use the fewest columns necessary for uniqueness

**Examples**:
- Employee ID in an employee table
- Social Security Number in a citizen database
- Order ID in an orders table
- Student ID in an academic system

```sql
-- Single column primary key
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

-- Composite primary key (multiple columns)
CREATE TABLE Order_Items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

#### Unique Key
**Definition**: A unique key constraint ensures that all values in a column or group of columns are distinct across all rows in the table. Unlike primary keys, unique keys allow one NULL value and a table can have multiple unique key constraints.

**Characteristics**:
- **Distinctness**: All non-NULL values must be unique
- **NULL allowance**: Can contain one NULL value
- **Multiple constraints**: A table can have multiple unique keys
- **Index creation**: Automatically creates a unique index

**Use Cases**:
- Email addresses in user registration systems
- Phone numbers in contact databases
- Product codes in inventory systems
- License plate numbers in vehicle databases

```sql
CREATE TABLE Users (
    user_id INT PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(15) UNIQUE
);
```

#### Foreign Key
**Definition**: A foreign key is a column or set of columns in one table that refers to the primary key in another table. It establishes and enforces a link between the data in two tables, maintaining referential integrity.

**Characteristics**:
- **Referential Integrity**: Values must exist in the referenced table or be NULL
- **Relationship Establishment**: Creates logical connections between tables
- **Cascade Options**: Can define actions when referenced data is modified or deleted
- **Multiple References**: Can reference different tables or the same table (self-referencing)

**Cascade Options**:
- **CASCADE**: Automatically delete/update dependent records
- **SET NULL**: Set foreign key to NULL when referenced record is deleted
- **RESTRICT**: Prevent deletion/update if dependent records exist
- **NO ACTION**: Similar to RESTRICT but check is deferred

**Examples**:
```sql
-- Basic foreign key
CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

-- Foreign key with cascade options
CREATE TABLE Order_Items (
    item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES Products(product_id) ON DELETE RESTRICT
);
```

---

## Data Manipulation (CRUD Operations)

### Insert Data (Single Row)
```sql
INSERT INTO table_name VALUES (1, "string1", "string2");
```

### Insert Data (Specific Columns)
```sql
INSERT INTO Students (name, email, age)
VALUES ('John Doe', 'john@email.com', 20);
```

### Insert Multiple Rows
```sql
INSERT INTO Students (name, email, age) VALUES
('Alice Smith', 'alice@email.com', 22),
('Bob Johnson', 'bob@email.com', 21);
```

### Update Data
```sql
UPDATE table_name
SET col_2 = 'new_value'
WHERE col_1 = 1;
```

### Delete Specific Data
```sql
DELETE FROM table_name WHERE col_1 = 1;
```

**Warning**: Always use WHERE clause with UPDATE and DELETE to avoid modifying all rows!

---

## SQL Clauses

### SELECT
Used to retrieve data from one or more tables in a database.
```sql
SELECT column1, column2, ... FROM table_name;
```

### FROM
Specifies the table from which to retrieve or delete data.
```sql
SELECT * FROM employees;
```

### WHERE
Filters records that meet a specific condition.
```sql
SELECT * FROM table_name WHERE Age >= 18;
```

### GROUP BY
Groups rows that have the same values in specified columns.
```sql
SELECT column, AGGREGATE_FUNCTION(column) FROM table GROUP BY column;
```

### HAVING
Filters groups created by GROUP BY based on a condition.
```sql
SELECT column, AGGREGATE_FUNCTION(column) FROM table GROUP BY column HAVING condition;
```

### ORDER BY
Sorts the result set by one or more columns.
```sql
SELECT column1, column2 FROM table ORDER BY column1 ASC|DESC;
```

### LIMIT / TOP
**LIMIT** (MySQL, PostgreSQL):
```sql
SELECT * FROM orders LIMIT 5;
```

**TOP** (SQL Server):
```sql
SELECT TOP 5 * FROM orders;
```

---

## SQL Operators

### Arithmetic Operators
| Operator | Description | Example |
|----------|-------------|---------|
| + | Addition | SELECT 5 + 3; |
| - | Subtraction | SELECT 5 - 3; |
| * | Multiplication | SELECT 5 * 3; |
| / | Division | SELECT 5 / 3; |
| % | Modulo | SELECT 5 % 3; |

### Comparison Operators
| Operator | Description |
|----------|-------------|
| = | Equal to |
| !=, <> | Not equal to |
| > | Greater than |
| < | Less than |
| >= | Greater than or equal to |
| <= | Less than or equal to |

### Logical Operators
| Operator | Description |
|----------|-------------|
| AND | Returns true if both conditions are true |
| OR | Returns true if either condition is true |
| NOT | Returns true if condition is false |
| IN | Returns true if value is in a list |
| BETWEEN | Returns true if value is within a range |
| LIKE | Returns true if value matches a pattern |

---

## CASE Statements

The CASE statement enables conditional logic directly within queries, allowing dynamic query outputs based on conditions.

```sql
SELECT
    emp_id,
    salary,
    CASE
        WHEN salary > 100000 THEN 'High'
        WHEN salary BETWEEN 50000 AND 100000 THEN 'Medium'
        ELSE 'Low'
    END AS Salary_Level
FROM employees;
```

---

## Indexes

An index in SQL is a database object that improves the speed of data retrieval operations on a table.

### Create Index
```sql
CREATE INDEX IX_EmpID ON Sudish_Info(Emp_Id);
```

### Benefits of Indexes:
- Speeds up data retrieval using SELECT queries
- Improves performance of WHERE, JOIN, ORDER BY, and GROUP BY clauses
- Reduces disk I/O and CPU usage by narrowing data scans
- Supports data integrity through UNIQUE indexes
- Enhances sorting and filtering for reports and dashboards
- Optimizes performance on large tables with millions of rows

---

## Aggregate Functions

Aggregate functions process multiple rows of data and return a single summarized result.

### Common Aggregate Functions:
```sql
-- COUNT(): Counts the number of rows
SELECT COUNT(column_name) FROM table_name;

-- SUM(): Adds up all values in a numeric column
SELECT SUM(Salary) FROM employees;

-- AVG(): Calculates the average of numeric columns
SELECT AVG(Age) FROM employees;

-- MIN(): Finds the smallest value in the column
SELECT MIN(Salary) FROM employees;

-- MAX(): Finds the largest value in the column
SELECT MAX(Salary) FROM employees;
```

---

## JOINS

A JOIN combines rows from two or more tables based on a related column between them.

### Sample Tables Setup
```sql
-- Employees Table
CREATE TABLE Employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(100),
    dept_id INT
);

-- Sample Data
INSERT INTO Employees VALUES 
(1, 'John Doe', 101),
(2, 'Jane Smith', 102),
(3, 'Bob Wilson', NULL);

-- Departments Table
CREATE TABLE Departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(100)
);

-- Sample Data
INSERT INTO Departments VALUES 
(101, 'Engineering'),
(102, 'Marketing'),
(103, 'Sales');
```

### Types of Joins:

#### INNER JOIN
Returns only rows where there is a match in both tables.
```sql
SELECT e.name, d.dept_name
FROM Employees e
INNER JOIN Departments d
ON e.dept_id = d.dept_id;
-- Result: John Doe - Engineering, Jane Smith - Marketing
```

#### LEFT JOIN
Returns all rows from the left table and matched rows from the right.
```sql
SELECT e.name, d.dept_name
FROM Employees e
LEFT JOIN Departments d
ON e.dept_id = d.dept_id;
-- Result: John Doe - Engineering, Jane Smith - Marketing, Bob Wilson - NULL
```

#### RIGHT JOIN
Returns all rows from the right table and matched rows from the left.
```sql
SELECT e.name, d.dept_name
FROM Employees e
RIGHT JOIN Departments d
ON e.dept_id = d.dept_id;
-- Result: John Doe - Engineering, Jane Smith - Marketing, NULL - Sales
```

#### FULL JOIN
Returns all records from both tables.
```sql
SELECT e.name, d.dept_name
FROM Employees e
FULL JOIN Departments d
ON e.dept_id = d.dept_id;
-- Result: John Doe - Engineering, Jane Smith - Marketing, Bob Wilson - NULL, NULL - Sales
```

---

## Subqueries

A subquery is a query nested inside another query.

### Scalar Subquery (Returns Single Value)
```sql
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### Row Subquery (Returns Single Row)
```sql
SELECT * FROM employees
WHERE (dept_id, salary) = (SELECT dept_id, MAX(salary)
                          FROM employees
                          WHERE dept_id = 101);
```

### Table Subquery (Returns Multiple Rows)
```sql
SELECT name FROM employees
WHERE dept_id IN (SELECT dept_id
                  FROM departments
                  WHERE location = 'New York');
```

**Performance Tip**: Sometimes JOINs perform better than subqueries. Test both approaches for large datasets!

---

## Window Functions

Window functions perform calculations across a set of rows related to the current row without grouping the result set.

### ROW_NUMBER()
```sql
SELECT name, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;
```

### RANK() and DENSE_RANK()
```sql
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;
```

### Partition By
```sql
SELECT name, dept_id, salary,
       AVG(salary) OVER (PARTITION BY dept_id) as dept_avg_salary
FROM employees;
```

### Real-world Use Case: Top 3 Highest-Paid Employees per Department
```sql
SELECT * FROM (
    SELECT name, dept_id, salary,
           ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) as rn
    FROM employees
) ranked
WHERE rn <= 3;
```

---

## Database Design & Normalization

Database normalization is the process of organizing data to reduce redundancy and improve data integrity.

### Normal Forms:

#### First Normal Form (1NF)
- Each column contains atomic (indivisible) values
- Each column contains values of the same type
- Each column has a unique name
- Order doesn't matter

#### Second Normal Form (2NF)
- Must be in 1NF
- No partial dependencies (non-key attributes depend on entire primary key)

#### Third Normal Form (3NF)
- Must be in 2NF
- No transitive dependencies (non-key attributes don't depend on other non-key attributes)

### Normalization Example:

#### Unnormalized Table (Violates 1NF):
| student_id | name | course1 | course2 | instructor1 | instructor2 |
|------------|------|---------|---------|-------------|-------------|
| 1 | John | Math | Physics | Dr. Smith | Dr. Jones |
| 2 | Alice | Chemistry | Biology | Dr. Brown | Dr. Davis |

#### Normalized Tables (3NF):

**Students Table:**
```sql
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100)
);
```

**Courses Table:**
```sql
CREATE TABLE Courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(100),
    instructor VARCHAR(100)
);
```

**Enrollments Table:**
```sql
CREATE TABLE Enrollments (
    student_id INT,
    course_id INT,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES Students(student_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);
```

---

## Views, Stored Procedures & Functions

### Views
A view is a virtual table based on the result of an SQL statement.

```sql
-- Create View
CREATE VIEW high_salary_employees AS
SELECT name, salary, dept_id
FROM employees
WHERE salary > 50000;

-- Use View
SELECT * FROM high_salary_employees
WHERE dept_id = 101;
```

### Stored Procedures
Stored procedures are prepared SQL code that can be saved and reused.

```sql
-- Create Stored Procedure
CREATE PROCEDURE GetEmployeesByDept(@dept_id INT)
AS
BEGIN
    SELECT name, salary
    FROM employees
    WHERE dept_id = @dept_id;
END;

-- Execute Stored Procedure
EXEC GetEmployeesByDept @dept_id = 101;
```

### Functions
Functions return a single value and can be used in SQL expressions.

```sql
-- Create Function
CREATE FUNCTION CalculateBonus(@salary DECIMAL(10,2))
RETURNS DECIMAL(10,2)
AS
BEGIN
    RETURN @salary * 0.10;
END;

-- Use Function
SELECT name, salary, dbo.CalculateBonus(salary) as bonus
FROM employees;
```

---

## Transactions & ACID Properties

A transaction is a sequence of operations performed as a single logical unit of work.

### ACID Properties:
- **Atomicity**: All operations succeed or all fail
- **Consistency**: Database remains in valid state
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed changes are permanent

### Transaction Control:
```sql
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- If both updates successful
COMMIT;

-- If any error occurs
-- ROLLBACK;
```

**Important**: Always use transactions for operations that modify multiple related records!

---

## Performance Optimization

### Query Optimization Tips:
- **Use indexes wisely**: Create indexes on frequently queried columns
- **Avoid SELECT ***: Select only needed columns
- **Use LIMIT/TOP**: Limit result sets when possible
- **Optimize JOIN conditions**: Use proper indexes on join columns
- **Use EXISTS instead of IN**: For better performance with subqueries

### Optimization Example:

#### Slow Query
```sql
SELECT * FROM orders o, customers c
WHERE o.customer_id = c.customer_id
AND c.city = 'New York';
```

#### Optimized Query
```sql
SELECT o.order_id, o.order_date, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE c.city = 'New York'
LIMIT 100;
```

### Execution Plans

#### MySQL/PostgreSQL
```sql
EXPLAIN SELECT * FROM employees WHERE dept_id = 101;
```

#### SQL Server
```sql
SET STATISTICS IO ON;
SELECT * FROM employees WHERE dept_id = 101;
```

---

## Real-world Project Examples

### E-commerce Database Schema

#### Create Tables
```sql
-- Customers Table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_date DATE DEFAULT GETDATE()
);

-- Products Table
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2),
    category VARCHAR(50)
);

-- Orders Table
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE DEFAULT GETDATE(),
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order Items Table
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10,2),
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

#### Sample Data

**Customers:**
| customer_id | name | email | created_date |
|-------------|------|-------|--------------|
| 1 | John Smith | john@email.com | 2024-01-15 |
| 2 | Sarah Wilson | sarah@email.com | 2024-02-20 |
| 3 | Mike Johnson | mike@email.com | 2024-03-10 |

**Products:**
| product_id | name | price | category |
|------------|------|-------|----------|
| 101 | Laptop | 999.99 | Electronics |
| 102 | Mouse | 25.50 | Electronics |
| 103 | Desk Chair | 149.99 | Furniture |
| 104 | Monitor | 299.99 | Electronics |

**Orders:**
| order_id | customer_id | order_date | total_amount |
|----------|-------------|------------|--------------|
| 1001 | 1 | 2024-07-15 | 1025.49 |
| 1002 | 2 | 2024-07-20 | 449.98 |
| 1003 | 1 | 2024-08-01 | 149.99 |

**Order Items:**
| order_id | product_id | quantity | price |
|----------|------------|----------|-------|
| 1001 | 101 | 1 | 999.99 |
| 1001 | 102 | 1 | 25.50 |
| 1002 | 103 | 1 | 149.99 |
| 1002 | 104 | 1 | 299.99 |
| 1003 | 103 | 1 | 149.99 |

### Business Queries

#### Top 5 Customers by Total Spending
```sql
SELECT c.name, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 5;
-- Result: John Smith - $1175.48, Sarah Wilson - $449.98
```

#### Monthly Sales Report
```sql
SELECT
    YEAR(order_date) as year,
    MONTH(order_date) as month,
    COUNT(*) as total_orders,
    SUM(total_amount) as revenue
FROM orders
GROUP BY YEAR(order_date), MONTH(order_date)
ORDER BY year DESC, month DESC;
-- Result: 2024-08: 1 orders, $149.99; 2024-07: 2 orders, $1475.47
```

#### Products Never Ordered
```sql
SELECT p.name
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.product_id IS NULL;
```

---

## Practice Resources

### Learning Platforms
- **W3Schools SQL Tutorial**: https://www.w3schools.com/sql/
- **LeetCode SQL 50**: https://leetcode.com/studyplan/top-sql-50/
- **HackerRank SQL Domain**: https://www.hackerrank.com/domains/sql

### Software for Hands-on Practice
- **SSMS** - SQL Server Management Studio

### Project Ideas for Practice
1. Library Management System
2. Hospital Management System  
3. Student Information System
4. Inventory Management System
5. Social Media Database

---

## Best Practices & Tips

1. **Always backup** your database before making changes, especially with DROP or DELETE commands
2. **Always use WHERE clause** with UPDATE and DELETE to avoid modifying all rows
3. **Use indexes wisely** on frequently queried columns
4. **Avoid SELECT *** - select only needed columns
5. **Use transactions** for operations that modify multiple related records
6. **Test with EXPLAIN** to understand query performance
7. **Use meaningful names** for tables, columns, and constraints
8. **Follow normalization rules** to reduce data redundancy
9. **Use constraints** to maintain data integrity
10. **Document your database schema** and complex queries

### Performance Tips
- Sometimes JOINs perform better than subqueries for large datasets
- Use EXISTS instead of IN for better performance with subqueries
- Create functional, working demonstrations rather than placeholders
- Test both approaches and measure performance differences

## Frequently Asked Questions (FAQs)

### 1. What is the difference between SQL and MySQL?
**Answer**: SQL (Structured Query Language) is a standard programming language used to manage relational databases, while MySQL is a specific database management system that implements SQL. Think of SQL as the language and MySQL as one of many "dialects" or implementations (others include PostgreSQL, Oracle, SQL Server).

### 2. Can a table have multiple primary keys?
**Answer**: No, a table can have only one primary key. However, a primary key can be composite, meaning it consists of multiple columns combined together to create a unique identifier for each row.

### 3. What happens if I try to insert a duplicate value into a primary key column?
**Answer**: The database will reject the insertion and return an error. Primary key constraints prevent duplicate values to maintain data integrity and ensure each record can be uniquely identified.

### 4. When should I use INNER JOIN vs LEFT JOIN?
**Answer**: Use INNER JOIN when you only want records that have matching values in both tables. Use LEFT JOIN when you want all records from the left table, even if there are no matches in the right table (unmatched records will show NULL values for right table columns).

### 5. What is the difference between DELETE and TRUNCATE?
**Answer**: DELETE removes specific rows based on conditions and can be rolled back, while TRUNCATE removes all rows from a table, is faster, uses less transaction log space, but cannot be rolled back in most databases.

### 6. How do I prevent SQL injection attacks?
**Answer**: Use parameterized queries or prepared statements instead of concatenating user input directly into SQL strings. Also validate and sanitize all user inputs, use stored procedures when possible, and implement proper access controls.

### 7. What is the difference between CHAR and VARCHAR data types?
**Answer**: CHAR is fixed-length and always uses the specified number of characters (padding with spaces if necessary), while VARCHAR is variable-length and only uses as much storage as needed for the actual data. CHAR is faster for fixed-size data, VARCHAR is more storage-efficient.

### 8. Can I use ORDER BY with columns not in the SELECT clause?
**Answer**: Yes, in most cases you can ORDER BY columns that are not in the SELECT list, as long as those columns exist in the table(s) you're querying from. However, when using DISTINCT or GROUP BY, some databases may have restrictions.

### 9. What is the purpose of indexing and when should I create indexes?
**Answer**: Indexes improve query performance by creating fast access paths to data, similar to a book's index. Create indexes on columns frequently used in WHERE clauses, JOIN conditions, and ORDER BY clauses. However, avoid over-indexing as it can slow down INSERT, UPDATE, and DELETE operations.

### 10. What is the difference between HAVING and WHERE clauses?
**Answer**: WHERE filters rows before grouping occurs and cannot use aggregate functions, while HAVING filters groups after GROUP BY and can use aggregate functions. Use WHERE for row-level filtering and HAVING for group-level filtering.

### 11. How do I handle NULL values in SQL queries?
**Answer**: Use IS NULL or IS NOT NULL for checking NULL values (never use = NULL). Use COALESCE() or ISNULL() functions to provide default values for NULLs. Remember that most operations with NULL return NULL.

### 12. What is the difference between UNION and UNION ALL?
**Answer**: UNION removes duplicate rows from the combined result set, while UNION ALL includes all rows even if they're duplicates. UNION ALL is faster because it doesn't need to check for duplicates.

### 13. How do I optimize slow-running queries?
**Answer**: Use EXPLAIN to analyze query execution plans, add appropriate indexes, avoid SELECT *, use specific column names, optimize JOIN conditions, consider query rewriting, and ensure statistics are up to date.

### 14. What is database normalization and why is it important?
**Answer**: Normalization is the process of organizing data to reduce redundancy and improve data integrity. It eliminates duplicate data, ensures data consistency, makes updates easier, and reduces storage requirements. However, highly normalized databases may require more complex queries.

### 15. Can I modify data in a VIEW?
**Answer**: It depends on the view definition. Simple views based on a single table without complex operations can often be updated. Views with JOINs, aggregates, DISTINCT, or calculated fields are typically read-only. Check your specific database system's documentation for detailed rules.

### 16. What is the difference between a correlated and non-correlated subquery?
**Answer**: A non-correlated subquery can run independently and executes once for the entire outer query. A correlated subquery references columns from the outer query and executes once for each row processed by the outer query, making it potentially slower.

### 17. How do I choose between VARCHAR(255) and TEXT data types?
**Answer**: Use VARCHAR(255) for shorter strings with a known maximum length where you need indexing and fast searching. Use TEXT for longer, variable-length content where the size is unpredictable or very large, but note that TEXT columns may have indexing limitations.

### 18. What is the difference between a clustered and non-clustered index?
**Answer**: A clustered index physically reorders the table data and there can be only one per table (usually on the primary key). Non-clustered indexes create separate structures that point to the actual data rows, and you can have multiple non-clustered indexes per table.

### 19. How do I handle concurrent access to prevent data conflicts?
**Answer**: Use transactions with appropriate isolation levels, implement optimistic or pessimistic locking strategies, use timestamps or version numbers for conflict detection, and design your application to handle deadlocks and lock timeouts gracefully.

### 20. What is the difference between stored procedures and functions?
**Answer**: Stored procedures can perform multiple operations, modify data, and don't need to return a value, while functions must return a single value and typically cannot modify data. Functions can be used in SELECT statements, but stored procedures cannot.

### 21. How do I backup and restore a database?
**Answer**: Use database-specific backup commands (like BACKUP DATABASE in SQL Server or mysqldump in MySQL), schedule regular automated backups, test restore procedures regularly, and consider both full and incremental backup strategies based on your recovery requirements.

### 22. What are the ACID properties and why are they important?
**Answer**: ACID stands for Atomicity (all or nothing), Consistency (valid state), Isolation (concurrent transactions don't interfere), and Durability (committed changes persist). These properties ensure reliable transaction processing and data integrity in databases.

### 23. How do I design an efficient database schema?
**Answer**: Follow normalization principles to reduce redundancy, choose appropriate data types, establish proper relationships with foreign keys, plan for scalability, consider query patterns when designing indexes, and document your schema design decisions.

### 24. What is the difference between a database and a data warehouse?
**Answer**: A database is designed for transactional operations (OLTP) with frequent inserts, updates, and deletes, while a data warehouse is optimized for analytical queries (OLAP) with large amounts of historical data, typically updated in batches and optimized for complex reporting and analysis.

### 25. How do I migrate data from one database system to another?
**Answer**: Plan the migration carefully by analyzing schema differences, export data using standard formats (CSV, SQL dumps), handle data type conversions, test with sample data first, plan for minimal downtime, and always have a rollback strategy. Consider using specialized migration tools for complex scenarios.