
## What is SQL Injection ?

- SQL injection is a code injection technique that might destroy your database.
- SQL injection is one of the most common web hacking techniques.
- SQL injection is the placement of malicious code in SQL statements, via web page input.
# Example :
----------------------------------------------------------------------------------------------------
txtUserId = getRequestString("UserId"); \

txtSQL = "SELECT * FROM Users WHERE UserId = " + txtUserId; \

Ref : https://www.w3schools.com/sql/sql_injection.asp
------------------------------------------------------------------------------------------------------

## What is NTP amplification attack

An NTP amplification attack is a type of Distributed Denial of Service (DDoS) attack. In this attack, the attacker uses publicly accessible Network Time Protocol (NTP) servers 
to overwhelm a target system with UDP traffic. NTP is used by machines connected to the Internet to set their clocks accurately.

In an NTP amplification attack, the attacker sends short requests to an open NTP server. The response from the NTP server is dozens of times larger than the request. 
This is called the amplification effect. 
