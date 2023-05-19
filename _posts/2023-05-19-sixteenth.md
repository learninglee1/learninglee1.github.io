---
layout: single
title:  "Python 기초(16)"
---

# Python

#### 기본형


```python
# 정수형
# 수치 표현이 정수인 경우, 이를 대입한 변수는 자동으로 정수형이 됨
a = 1

# 부동소수점 수형
# 수치 표현에 소수점이 포함되면, 이를 대입한 변수는 자동적으로 부동소수점 수형이 됨
b = 2.0

# 문자열형
# 문자열은 싱글 쿼트(')로 감싸서 표현함
# 또는 더블 쿼트(")로 감싸도 상관 없음
c = 'abc'

# 논리형
# True 또는 False 중 하나를 취하는 변수형
d = True
```

#### print 함수와 type 함수


```python
# 정수형 변수 a의 값과 형
print(a)
print(type(a))
```

    1
    <class 'int'>
    


```python
# 부동소수점 수형 변수 b의 값과 형
print(b)
print(type(b))
```

    2.0
    <class 'float'>
    


```python
# 문자열형 변수 c의 값과 형
print(c)
print(type(c))
```

    abc
    <class 'str'>
    


```python
# 논리형 변수 d의 값과 형
print(d)
print(type(d))
```

    True
    <class 'bool'>
    

#### 2항연산


```python
# 덧셈
a1 = 1 + 2
print(a1)

# 뺄셈
a2 = 3 - 2
print(a2)

# 곱셈
a3 = 3 * 5
print(a3)

# 나눗셈
a4 = 13 / 5
print(a4)

# 나머지 연산
a5 = 13 % 5
print(a5)

# 몫 연산
a6 = 13 // 5
print(a6)

# 거듭 제곱
a7 = 2 ** 5
print(a7)
```

#### 논리 연산


```python
t1 = True
t2 = True
f1 = False
f2 = False

# AND 연산 
b1 = t1 and t2
print(b1)

# OR 연산
b2 = t1 or f1
print(b2)

# NOT 연산
b3 = not f1
print(b3)
```

#### 대입 연산


```python
c1 = 5

# c1 = c1 + 2와 같음
c1 += 2
print(c1)

# c1 = c1 - 3과 같음
c1 -= 3
print(c1)
```

#### 리스트


```python
# 리스트의 정의
l = [1, 2, 3, 5, 8, 13]

# 리스트의 값 과 형
print(l)
print(type(l))
```

    [1, 2, 3, 5, 8, 13]
    <class 'list'>
    

#### 리스트의 요소 수


```python
# 리스트의 요소 수
print(len(l))
```

    6
    

#### 리스트의 요소 참조


```python
# 리스트의 요소 참조

# 가장 첫번째 요소
print(l[0])

# 3번째 요소
print(l[2])

# 마지막 요소(이와 같은 지정 방식도 가능함)
print(l[-1])
```

    1
    3
    13
    

#### 부분 리스트 참조 1


```python
# 부분 리스트, 인덱스 : 2 이상 인덱스 : 5 미만
print(l[2:5])

# 부분 리스트, 인덱스 : 0 이상 인덱스 : 3 미만
print(l[0:3])

# 시작하는 인덱스가 0인 경우, 생략 가능
print(l[:3])
```

    [3, 5, 8]
    [1, 2, 3]
    [1, 2, 3]
    

#### 부분 리스트 참조  2


```python
# 부분 리스트, 인덱스 : 4 이상 마지막 까지
# 리스트의 길이(요소 수)를 구함
n = len(l) 
print(l[4:n])

# 마지막 요소는 인덱스 생략 가능
print(l[4:])

# 마지막에서 2개 요소
print(l[-2:])

# 처음과 마지막 인덱스를 생략하면, 리스트 전체를 참조
print(l[:])
```

    [8, 13]
    [8, 13]
    [8, 13]
    [1, 2, 3, 5, 8, 13]
    

#### 튜플


```python
# 튜플의 정의
t = (1, 2, 3, 5, 8, 13)

# 튜플의 값 출력
print(t)

# 튜플의 형 출력
print(type(t))

# 튜플의 요소 수
print(len(t))

# 튜플의 요소 참조
print(t[1])
```

    (1, 2, 3, 5, 8, 13)
    <class 'tuple'>
    6
    2
    


```python
t[1] = 1
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-12-d6b0ce29b2aa> in <module>()
    ----> 1 t[1] = 1
    

    TypeError: 'tuple' object does not support item assignment



```python
x = 1
y = 2
z = (x, y)
print(type(z))
```

    <class 'tuple'>
    


```python
a, b = z
print(a)
print(b)
```

    1
    2
    

### 사전

#### 사전의 정의


```python
# 사전의 정의
my_dict = {'yes': 1, 'no': 0}

# print문 출력 결과
print(my_dict)

# type 함수 출력 결과
print(type(my_dict))
```

    {'yes': 1, 'no': 0}
    <class 'dict'>
    

#### 사전의 참조


```python
# 키로부터 값을 참조

# key= 'yes' 로 검색
value1 = my_dict['yes']
print(value1)

# key='no' 로 검색
value2 = my_dict['no']
print(value2)
```

    1
    0
    

#### 사전에 새로운 항목 추가


```python
# 사전에 새로운 항목 추가
my_dict['neutral'] = 2

# 결과 확인
print(my_dict)
```

    {'yes': 1, 'no': 0, 'neutral': 2}
    

### 제어 구조

#### 루프 처리


```python
# 루프 처리

# 리스트 정의
list4 = ['One', 'Two', 'Three', 'Four']

# 루프 처리
for item in list4:
    print(item)
```

    One
    Two
    Three
    Four
    


```python
# range 함수를 사용한 루프 처리

for item in range(4):
    print(item)
```

    0
    1
    2
    3
    


```python
# 2개의 인수를 취하는 range 함수

for item in range(1, 5):
    print(item)
```


```python
# 사전과 루프 처리

# items 함수
print(my_dict.items())

# items 함수를 사용한 루프 처리

for key, value in my_dict.items():
    print(key, ':', value )
```

    dict_items([('yes', 1), ('no', 0), ('neutral', 2)])
    yes : 1
    no : 0
    neutral : 2
    

#### 조건 분기(if 문)


```python
# if 문 예시
for i in range(1, 5):
    if i % 2 == 0:
        print(i, '  짝수입니다')
    else:
        print(i, '  홀수입니다')
```

    1   홀수입니다
    2   짝수입니다
    3   홀수입니다
    4   짝수입니다
    

####  함수


```python
# 함수의 정의 예시 1
def square(x):
    p2 = x * x
    return p2

# 함수의 호출 예시 1
x1 = 13
r1 = square(x1)
print(x1, r1)
```

    13 169
    


```python
#  함수의 정의 예시 2
def squares(x):
    p2 = x * x
    p3= x * x * x
    return (p2, p3)

# 함수의 호출 예시 2
x1 = 13
p2, p3 = squares(x1)
print(x1, p2, p3)
```

    13 169 2197
    

#### 라이브러리 설치


```python
# 라이브러리 설치
!pip install matplotlib | tail -n 1
```

    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib) (1.15.0)
    

#### import 문


```python
# 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 데이터 프레임 표시 함수
from IPython.display import display
```

#### warning을 출력하지 않는 방법


```python
# 필요하지 않은 warning 출력하지 않기
import warnings
warnings.filterwarnings('ignore')
```

#### 수치의 출력 형식 지정


```python
# f 문자열 표시
a1 = 1.0/7.0
a2 = 123

str1 = f'a1 = {a1}   a2 = {a2}'
print(str1)
```

    a1 = 0.14285714285714285   a2 = 123
    


```python
# f 문자열의 상세 옵션

# .4f : 소수점 이하 네 자리 고정소수점 표시
# 04 : 정수를 0을 포함해 네 자리까지 표시
str2 = f'a1 = {a1:.4f}  a2 = {a2:04}'
print(str2)

# 04e : 소수점 이하 네 자리 부동소수점 표시
# #x : 정수를 16진수로 표시
str3 = f'a1 = {a1:.04e}  a2 = {a2:#x}'
print(str3)
```

    a1 = 0.1429  a2 = 0123
    a1 = 1.4286e-01  a2 = 0x7b
    
