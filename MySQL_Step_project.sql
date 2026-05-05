-- SQL степ-проект 
-- Запити
-- 1. Покажіть середню зарплату співробітників за кожен рік, до 2002 року.
SELECT ROUND(AVG(salary), 2) AS salary, YEAR(from_date) AS y
FROM salaries
WHERE YEAR(from_date) <= 2002
GROUP BY y
ORDER BY y;


-- 2. Покажіть середню зарплату співробітників по кожному відділу.
-- Примітка: потрібно розрахувати по поточній зарплаті, та поточному відділу співробітників
SELECT de.dept_no,  AVG(salary) AS salary
FROM dept_emp AS de
INNER JOIN salaries AS s ON (de.emp_no = s.emp_no AND CURRENT_DATE() BETWEEN s.from_date AND s.to_date)
WHERE CURRENT_DATE() BETWEEN de.from_date AND de.to_date
GROUP BY de.dept_no
ORDER BY de.dept_no;


-- 3. Покажіть середню зарплату співробітників по кожному відділу за кожний рік
SELECT de.dept_no, ROUND(AVG(salary), 2) AS salary, YEAR(de.from_date) AS y_r
FROM salaries AS s
INNER JOIN dept_emp AS de ON (s.emp_no = de.emp_no)
GROUP BY de.dept_no, y_r
ORDER BY de.dept_no, y_r;


-- 4. Покажіть відділи в яких зараз працює більше 15000 співробітників.
SELECT d.dept_name, COUNT(DISTINCT de.emp_no) AS emp
FROM dept_emp AS de
INNER JOIN departments AS d ON (d.dept_no = de.dept_no
		AND CURRENT_DATE() BETWEEN de.from_date AND de.to_date)
GROUP BY dept_name
HAVING emp > 15000
ORDER BY emp;

-- 5. Для менеджера який працює найдовше покажіть його номер, відділ, дату прийому на роботу, прізвище
SELECT e.emp_no, dm.dept_no, e.hire_date, e.last_name
FROM dept_manager AS dm
INNER JOIN employees AS e ON (e.emp_no = dm.emp_no AND CURRENT_DATE() BETWEEN dm.from_date AND dm.to_date)
ORDER BY DATEDIFF(CURRENT_DATE(), e.hire_date) DESC LIMIT 1;

-- 6. Покажіть топ-10 діючих співробітників компанії з найбільшою різницею між їх зарплатою і середньою зарплатою в їх відділі.
WITH s_emp AS (SELECT s.emp_no, s.salary, de.dept_no FROM salaries AS s INNER JOIN dept_emp AS de ON (s.emp_no = de.emp_no)  
				WHERE CURRENT_DATE() BETWEEN s.from_date AND s.to_date AND CURRENT_DATE() BETWEEN de.from_date AND de.to_date),  -- по співробітнику
	a_de AS (SELECT d.dept_no, AVG(salary) AS a_v FROM salaries AS sl INNER JOIN dept_emp AS d ON (sl.emp_no = d.emp_no)   
				WHERE CURRENT_DATE() BETWEEN sl.from_date AND sl.to_date AND CURRENT_DATE() BETWEEN d.from_date AND d.to_date
				GROUP BY d.dept_no), -- -- по середній зарплаті відділу 
   e_diff AS (SELECT s_emp.emp_no, s_emp.dept_no, s_emp.salary, a_de.a_v, s_emp.salary - a_de.a_v AS diff_f FROM s_emp INNER JOIN a_de ON (s_emp.dept_no = a_de.dept_no))
SELECT *
FROM e_diff
ORDER BY diff_f DESC LIMIT 10;


-- 7. Для кожного відділу покажіть другого по порядку менеджера. Необхідно вивести відділ, 
-- прізвище ім’я менеджера, дату прийому на роботу менеджера і дату коли він став менеджером відділу
WITH m_d AS (SELECT dm.emp_no, dm.dept_no, dm.from_date, ROW_NUMBER() OVER (PARTITION BY dm.dept_no ORDER BY dm.from_date) AS m
FROM dept_manager AS dm)
SELECT m_d.dept_no, e.last_name, e.first_name, e.hire_date, m_d.from_date
FROM m_d
INNER JOIN employees AS e ON (e.emp_no = m_d.emp_no)
WHERE m = 2;

-- База даних
DROP DATABASE IF EXISTS course_management;
CREATE DATABASE IF NOT EXISTS course_management;
USE course_management;


DROP TABLE IF EXISTS courses;
CREATE TABLE IF NOT EXISTS courses (
course_no INT AUTO_INCREMENT PRIMARY KEY,
course_name VARCHAR(40) NOT NULL,
start_date DATE NOT NULL,
end_date DATE NOT NULL
);

DROP TABLE IF EXISTS teachers;
CREATE TABLE IF NOT EXISTS teachers (
teacher_no INT AUTO_INCREMENT PRIMARY KEY,
teacher_name VARCHAR(30) NOT NULL,
phone_no VARCHAR(30) NOT NULL
);

DROP TABLE IF EXISTS students;
CREATE TABLE IF NOT EXISTS students (
student_no INT AUTO_INCREMENT,
course_no INT NOT NULL,
teacher_no INT NOT NULL,
student_name VARCHAR(30) NOT NULL,
email VARCHAR(60) NOT NULL,
birth_date DATE NOT NULL,
PRIMARY KEY (student_no , course_no, teacher_no)
,FOREIGN KEY (course_no)
REFERENCES courses (course_no)
ON UPDATE RESTRICT ON DELETE CASCADE, 
FOREIGN KEY (teacher_no)
REFERENCES teachers (teacher_no)
ON UPDATE RESTRICT ON DELETE CASCADE
);

-- DESC students;

START TRANSACTION;

INSERT INTO courses (course_name, start_date, end_date)
VALUES ('Data Analyst', '2025-01-15', '2025-06-12'),
	('Data Science', '2025-02-21', '2025-09-25'),
	('Data Engineer', '2025-03-08', '2025-10-12'),
	('Front End', '2025-04-22', '2025-10-24'),
	('Fullstack Node.js', '2025-05-10', '2025-12-12'),
	('DevOps', '2025-06-24', '2026-02-12'),
	('Java', '2025-07-07', '2026-02-20')
	;
    
INSERT INTO teachers (teacher_name, phone_no)
VALUES ('John', '+38011111111'),
	('John', '+38022222222'),
	('Oleg', '+38033333333'),
	('Emma', '+38044444444'),
	('Nastya', '+38055555555'),
	('Bella', '+38066666666'),
	('Roman', '+38077777777'),
	('Dmytro', '+38088888888')
	;
    
INSERT INTO students(course_no, teacher_no, student_name, email, birth_date)
VALUES (1, 1, 'Bogdan', 'aaaaa.1@gmail.com', '2000-01-10'),
(2, 2, 'Anatoliy', 'aaaaa.2@gmail.com', '2000-01-01'),
(3, 3, 'Alla', 'aaaaa.3@gmail.com', '2000-01-02'),
(3, 4, 'Tatyana', 'aaaaa.4@gmail.com', '2000-01-03'),
(4, 5, 'Mila', 'aaaaa.5@gmail.com', '2000-01-04'),
(5, 6, 'Olga', 'aaaaa.6@gmail.com', '2000-01-05'),
(6, 7, 'Inna', 'aaaaa.7@gmail.com', '2000-01-06'),
(7, 8, 'Volodymyr', 'aaaaa.8@gmail.com', '2000-01-07'),
(6, 7, 'Igor', 'aaaaa.9@gmail.com', '2000-01-08'),
(1, 1, 'Nina', 'aaaaa.10@gmail.com', '2000-01-09'); 

COMMIT;  

-- SELECT * FROM courses;
-- SELECT * FROM teachers;
-- SELECT * FROM students;

-- 3. По кожному викладачу покажіть кількість студентів з якими він працював( Уточнити?)
SELECT st.teacher_no, t.teacher_name , COUNT(student_no) AS count_st
FROM teachers AS t
INNER JOIN students AS st ON (t.teacher_no = st.teacher_no)
INNER JOIN courses AS cs ON (cs.course_no = st.course_no) 
GROUP BY t.teacher_no;


-- 4. Спеціально зробіть 3 дубляжі в таблиці students (додайте ще 3 однакові рядки)
INSERT INTO students (course_no, teacher_no, student_name, email, birth_date)
SELECT course_no, teacher_no, student_name, email, birth_date
FROM students LIMIT 3;

-- 5. Напишіть запит який виведе дублюючі рядки в таблиці students
SELECT student_name, email, COUNT(*)
FROM students
GROUP BY email, student_name
HAVING COUNT(*) > 1;