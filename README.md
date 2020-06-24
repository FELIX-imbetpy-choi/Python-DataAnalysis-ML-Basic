# Python-DataAnalysis-ML-Basic
데이터 분석을 위해 Python 을 활용하여 기초 공부를 진행하였습니다.

Python 경로찾기


[02.자료형] 정리완료
Count – 문자열 개수
Find – 문자열에서 문자 찾기
Index – 문자위치 알려주기
Join  - 삽입  
Upper – 대문자
Lower – 소문자
Lstrip – 왼쪽공백제거
Rsttrip – 오른쪽공백제거
Strip – 양쪽공백제거
Replace – 문자열 수정 바꾸기
Split – 문자열 나누기
Format – 고급문자열 사용
#2개 이상의 값 넣기
number = 10
day = 'three'
'I have {0} apples, I will go abroad for {1} days'.format(number, day)

[03.리스트자료형] 정리완료
List
Append – 리스트에 추가하기
Sort – 정렬
Reverse – 역순정렬
Insert – 리스트에 요소 삽입
Remove – 리시트에 값 제거
Pop – 리스트 마지막을 제거하고 보여줌
Len() – 리스트의 개수를 보여줌

[04.튜플자료형] 정리완료

[05.딕셔너리] 정리완료
Dic[] 
Keys – 키값을 알려줌
Values – 값들을 알려줌
dic.items() – 키값과 값을 함께 얻기
clear – key와 value를 지우기.

[06.집합 자료형] 정리완료
& - 교집합
| - 차집합
Add – 값추가
Update – 값 여러 개 추가
Remove – 특정값 제거
Intersection – 교집합

[07.자료형의 참과 거짓] 정리완료
Del – 메모리에서 값 삭제
Copy – 리스트를 복사해서 쓸 수 있음
Is – 같냐고 물어보는 함수

[08.if문] 정리완료
[09.while문] 정리완료
[10.for문] 정리완료
[11.함수] 정리완료

[12.사용자의 입력과 출력] 정리완료
Input – 사용자 입력값
Open 파일 쓰고 읽기
readline() – 파일에서 한줄 읽어오기
readlines() – 한번에 모든 줄을 읽어와 리스트로 저장
read() – 한번에 모든 줄을 스트링으로 가져옴
write() – 파일에 쓰기를 실행
with문

[13.클래스] 정리완료
Class – 클래스 지정
Self – 클래스 사용시 변수사용

[14.모듈] 정리완료

[15.내장함수] 정리완료
Abs – 절대값
All – 하나라도 거짓이면 거짓
Any – 하나라도 참이면 참
Divmod – 몫과 나머지 튜플
Enumerate – 인덱스가 있는 형식으로 변환
List – 리스트형으로 변환
Max – 최댓값
Min – 최소값
Pow – (2,3) 2의 3제곱값
Str – 문자값으로 변환
Zip - 동일한 개수 자료형 묶기
Time.sleep – 쉬는시간을 두고 실행
Calendar – 달력을 불러온다
Shuffle – 섞기
Webbrowser – 사이트 열기

[16.Select DB] 정리완료
[17.insert DB] 정리완료
[18.updat DB] 정리완료
[19.DeleteDB] 정리완료
[20.동시실행] 정리완료

----------------------------------------기초구역--------------------------------------

[21.Pandas_Numpy] 정리완료
dataFrame – 데이터 형성
loc – 부분데이터 추출
Series - 1차원데이터는 이걸 사용
Dictionary - 키값을 주고 리스트 생성
Sort_values(by = ) – 기준으로 정렬시키기
Ascending – 내림차수능로 정렬
Numpy
# 딕셔너리 사용하여 데이터프레임 형성
Zeros – 0을 채움
Arange – 범위지정
Type – 해당하는 것의 데이터 타입확인

[22.분석맛보기] 정리완료
%matplotlib inline – 그림그리기 툴
Pd.read_csv() - csv파일 읽어오기(pandas로)
last_valid_index – 데이터 전체 개수 확인
pivot_table – 데이터 프레임 재형성
Tail – 뒤의 값부터 보여줘
Size – 데이터 몇 개인지
Count – 데이터 칼럼마다 개수
Names,columns – 컬럼 이름 바꾸기
# 정리된 data로 csv file 만들기 - DataFrame(names를 쓰는데/ 컬럼을 바꿔서 
names2 = pd.DataFrame(names,columns=['id','name','year','gender','births'])


[23.numpy_ndarray] 정리완료
Shape – 크기 확인
Array – 배열생성
Dtype – 데이터 타입 확인
#루트?
arr1 ** 0.5 – arr에 0.5

[24.array_indexing] 정리완료
2차원 배열
Random – 난수발생
np.random.randn(7,4) - 7행 4열의 가우시안 표준 정규분포 행열 생성
rand = 0부터 1까지 균일 분포
randn = 가우시안 표준 정규 분포
randint = 균일분포 정수 난수


[25. array_functuion] 정리완료
Sqrt – 루트값
Sin – 사인값
Seed – 난수값 고정 랜덤값 고정
Mean – 평균값
Axis = 0 열별로 합
Axis = 1 행별로 합
Unique = 중복값 제거

[26.Pandas_Titanic] 정리완료
Isin – 데이터 조건 검색시 2개이상의 조건 넣는 방법.
Shape – 데이터 구조 확인
train['Ticket'].str.contains('STON')– 문자열이 들어가 있는 데이터 추출/ 문자열 찾기
Isnull - null인 것을 true로 표시
Notnull – null이 아닌 것을 true 표시
~ - true와 false를 변경
train['Category'] = 'Titanic' 칼럼추가
# 승객 번호 1,3,5,13 을 출력
train.loc[[1,3,7,13]] – 인덱스 이용하여 출력
# loc사용시 행열에 해당하는 부분에 리스트를 주어서 사용 할 수 있음.
passenger_ids = [1,3,7,13]
train.loc[passenger_ids,:]


[27.  Pandas자료구조]  중요! 자료다루는 방법 다수 정리완료
Dtype – 데이터 타입 알아보기
describe() – 표준편차 확인
iloc
# Series에 index 적용
obj2 = pd.Series([4,7,-5,3],index=['d','b','a','c'])
# dictionary의 data를 Series로
sData = {'Charles':10000,'Kenny':20000,'Henry':30000}
obj3 = pd.Series(sData)
# data 이름을 설정
obj3.name = 'Salary'
# 인덱스 이름을 설정
obj3.index.name = 'name'
# 딕셔너리로 data frame 생성

#인덱스와 칼럼 정리
df.index.name = 'Num'
df.columns.name = 'Info'
# series 시리즈로 원하는 곳에 자료 삽입
val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
# 통계의 기초 통계량을 확인
df2.describe() # std = 표준편차(평균과의 차이값들의 평균) , 예측률, 오차범위, 신빙성 표준편차는 작을수록 좋음
# Series를 dataframe에 넣기
val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
df2['debt'] = val
#loc를 사용하여 names가 kenny인 정보중 names와 point만 추출하기
kenny = df.loc[:,'names']=='kenny'
df.loc[kenny,['names','points']]

[28.DataFrame 조작] 정리완료
Dropna( how = any)  - nan 값이 하나라도 있으면 행을 삭제.
Fillna – nan인 곳에 value를 삽입
Isnull – null 인곳을 true로 표시
df.drop(['B','D'],1) – 복수의 열 삭제
#특정 인덱스에 값 삽입
val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
#날짜값을 연속으로 인덱스 사용하기
df.index =  pd.date_range('20181116',periods=6)  # periods 몇일치 넣을꺼?
# 데이터에 nan null값 삽입하기
df['F'] = [1.0,np.nan,3.5,6.1,np.nan,7.0]
# nan이 있는 데이터 행 삭제 - any는 행데이터 중에 1개라도 nan이면 지운다 지워도 데이터는 그대로 있다!
df.dropna(how = 'any')
#nan을 특정값으로 변경 - nan값에 알아서 들어감
df.fillna(value = '0.5')
# null인 부분을 true 값으로 나타냄
Isnull
# 2개의 날짜 지우기!!!!(인덱스 통해서 지우기)
df.drop([pd.to_datetime('20181116'),pd.to_datetime('20181117')])
# 복수의 열 삭제 – drop 사용하기
df.drop(['B','D'],1)
# F열에서 nan을 포함하고 있는 행 찾기
df.loc[df.isnull()['F'],:]

[29.분석용 함수] 정리완료
# 행방향의 합계
df.sum(axis = 0)
# 열의 분산 구하기
df.var(axis = 0)
#사용자 함수 (lamda / 일반함수)
df['A'].corr(df['D']) – 상관계수
df['A'].cov(df['C']) – 분산

sort_index – 인덱스 오름차순 정리
# nan이 있는 곳을 계산하지 않는다.
df.mean(axis = 1, skipna=False)
#데이터 섞기
random_dates = np.random.permutation(dates) # 있는데이터를 섞어준다.
df2 = df.reindex(index = random_dates, columns=['D','B','C','A'])
#  index를 오름차순으로 정리
df2.sort_index(axis = 0)
# D열의 값을 기준으로 오름차순 정렬  - by D열기준으로!
df2.sort_values(by='D')
# one의 nan은 one의 평균값으로 대체한다.?
onenan = df['one'].isnull()
df[onenan] = df.loc[:,'one'].mean()
# two의 nan은 가장 작은 값으로 대체한다.
twonan = df['two'].isnull()
df[twonan] = df.loc[:,'two'].min()

apply – 함수 실행

[30. matplotlib function] 정리완료
Series
nbagg
line plot
Cumsum - 누적값
Df[‘B’].plot() – 특정한 열을 line plot으로 그려줌
Bar plot – 리스트로 인덱스 만들기 – list(‘abcd…..’)
.plot(kind = ‘bar’)
.plot(kind = ‘barh’)
concatenate
# 누적 막대 그래프
df2.plot(kind='bar', stacked = True)
# -1부터 1까지 랜덤,난수 발생
s3 = pd.Series(np.random.normal(0,1,size = 200))

# 히스토그램
s3.hist()
s3.hist(bins = 50) – bins 몇 개의 데이터를 나눠서 보여줄까?

# 산점도 그리기 - 산점도는 2개이상 못쓴다. - 뒤에선 파이썬에서만 3개까지 쓰는 법 배움
plt.scatter(df3['x1'],df3['x2'])


[31. Plot 모양 변경] 정리완료
# 하나의 figure에 여러개의  plot을 그리자
fig = plt.figure()
# plot 모양 꾸미기
plt.plot(np.random.randn(30),color = 'g', marker = 'o', linestyle = '-.')
alpha – 투명도 조절
ax.set_xticks – 수평축의 눈금을 다르게 변경
ax.set_xticklabels 
rotation – 라벨 기울기
fontsize – 글씨크기 조절
set_title – 그래프의 제목 설정
set_xlabel – x축 라벨
ax.legend(loc='best') -  범례를 니가 알아서 좋은위치에 놔라
set_xlim – 데이터 범위 변경

[32. 여러개의  DataFrame 합치기] 정리완료
http://rfriend.tistory.com/256 참고
pd.merge(df1,df2,on = 'key') – 키의 교집합
pd.merge(df1,df2,on = 'key', how = 'outer') – 키의 합집합
pd.merge(df1,df2,on = 'key', how = 'left') – 오른쪽 키 기준으로 나열
pd.merge(df3,df4,left_on = 'lkey', right_on = 'rkey') – 키가 다를경우 키지정하기

right1 = pd.DataFrame(  # 딕셔너리 데이터에 인덱스 구성하기.!
    {
        'group_val' : [3.5,7]
    }, index = ['a','b']
)

concat
reshape – 행렬을 조절 해줄 수 있음
5 + np.arange(4).reshape(2,2) – 0부터 시작되는 arange를 5부터 시작하게 함

[33.DataFrame 계층적 indexing] 정리완료



[34. DataFrame 데이터 변형하기] 정리완료
데이터 만들기 방식 another
df = pd.DataFrame(
    {
        'k1' : ['one'] * 3 + ['two'] * 4,
        'k2' : [1,1,2,3,3,4,4]
    }
)
df['v1'] = np.arange –새로운 열 추가
df.drop_duplicates(['k1','k2'],keep = 'last') – 특정열 중복값 제거, 뒤의 값으로 남기기
replace – 데이터 변경
df3['grade'] = df3['raw_grade'].astype('category') - 카테고리형으로 변환
# 데이터를 간단하게 변경시킬 수 있다.
df3['grade'].cat.categories = ['very good','good','very bad']
df3
categories – 데이터 조작에 수월함
split() – 문자를 띄어쓰기 기준으로 잘라줌 괄호안에 기준을 넣을 수 있다.

[35. 데이터 그룹화 함수] 정리완료
Groupby – 그룹화 해서 데이터 처리

[36. 웹의 다양한 데이터 형식] 정리완료
BeautifulSoup
urllib
os.path
JSON
import codecs
csv


[37.Data 분석 priview (Baby names )]
pivot_table
aggfuc – 그래프 안에서 쓸 수 있는 함수! Sum 등
groupby
subplots – 칼럼별 분할해서 그래프 그리기
reindex – 인덱스 값 변경
fig,axes = 
.subplots(2,1,1) 2행 1열의 그래프 를 첫번째에 그리기
.plot(kind='bar',ax=axes[0],title='Male') – 바로 그리고, 1번째에 그리고,제목은male
plt.tight_layout() – 그래프 레이아웃 자동조절

[39. Movie_Lens]
mean_rating_by_user_list = np.array(mean_rating_by_user_list,dtype = np.float32) – 리스트를 array 배열로 변환(관리 편의성)
np.savetxt('./data/mean_rating_by_user.csv',mean_rating_by_user_array, fmt='%.3f',delimiter=",") – fmt는 글씨 포맷 변경

[40. Lending Club]
dropna(how='any') – nan 값 제거 


[41.Game of Throne]

[42. 2016 US Election]
unstack() - 2차 인덱스를 빼낸다
agg

[42.서울시 인구대비 CCTV현황]
# 한글 폰트 문제 해결
Encoding - 한글 data는 항상 인코딩 타입을 설정해 주어야 한다.

inplace=True – 원데이터에 바로 적용 칼럼이름 변경 어디서든지 적용됨
usecols  -  는 엑셀의 칼럼순서이다.
.sort_values(by='소계',ascending=True) – sort  칼럼의 값별로

# 한글 폰트 문제 해결
grid=True – 보기편하게 선이 생김
figsize=(10,10) – 그래프 크기 / 그래프 그리기 전 설정해줘야.
plt.show() – 그래프 알아서 조절
plt.colorbar()


[43. 서울시 범죄 안전도]
위도 경도 뽑기
# 엑셀에서 숫자값에 ,이 붙은경우 숫자로 인식하도록 thousands 사용
thousands=','

[gu for gu in temp if gu[-1]=='구'][0] – 리스트에 for문 사용

# value값 바꿀때 loc사용
crime_anal_police.loc[crime_anal_police['관서명']=='금천서',['구별']]='금천구'

# pivot table  - 인덱스 변경하여 데이터 처리
crime_anal=pd.pivot_table(crime_anal_police, index='구별', aggfunc=np.sum)
#상관계수  중앙값 맞추기
nomalize

#칼럼 이름 바꾸기
crime_anal.rename(columns={
    '강간 발생' : '강간',
    '강도 발생' : '강도',
    '살인 발생' : '살인',
    '절도 발생' : '절도',
    '폭력 발생' : '폭력'},inplace = True)

# 데이터의 정규화 (Nomalization)

# Nomalize 하는 머신러닝
Sklearn	preprocessing

#제이슨 json 파일 지도 열기
geo_str = json.load(open(geo_path,encoding='utf-8'))

# 데이터 복사
crime_anal_raw = crime_anal_police.copy()

[44.Seaborn_그리기도구] 정리완료
#seaborn
import seaborn as sns

#seaborn 그리기
sns.set_style('darkgrid') # seaborn의 darkgrid 스타일 - 피규어 그리드를 쓰지 않아도 된다.
plt.plot(x,y1,x,y2,x,y3,x,y4)
plt.show

#샘플데이터 부르기
tips = sns.load_dataset('tips')
tips.head()

#색변경
palette='Set3'

# lmplot – 회귀선도 같이 보여준다.
sns.lmplot(x='total_bill',y='tip',data=tips,height=10) # 팁이 10달러인것 까지 보겠다?
plt.show()

# 연도별 월별 항공기 승객 수 구분 – 데이터 정제
flights = filghts.pivot('month','year','passengers') #pivot_table은 aggfunc 함수가 있어야 한다 없이 하려면 pivot

# Heatmap을 이용하여 승객수를 색상으로 표현하기.
plt.figure(figsize=(10,8))
sns.heatmap(flights,annot=True,fmt='d') #annot 숫자 보여주기(승객수보여주기) / fmt (f:float , d:digit)
plt.show()


[45.folium] 정리완료
# 위도와 경도 좌표 입력으로 해당 위경도에 맞는 지도 보여주기
map_osm = folium.Map(location=[41.37514507063049,2.1478888392448425],zoom_start=14) #바르셀로나

# 팝업(popup) – 클릭시 팝업 메시지 띄움
popup 

# json파일을 하용해 지도 정보를 추출하고 지도출력
Choropleth


[46. Scraping]
# library
import urllib.request
# URL과 저장경로 - url경로에 가서 savename으로 저장
url = "http://uta.pw/shodou/img/28/214.png"
savename = './data/test01.png'
# 다운로드 하기
urllib.request.urlretrieve(url,savename)
print('저장되었습니다.')


기상청
 "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

#기본 사용 import
from bs4 import BeautifulSoup


# strip()
공백 지우기


[47. Scraping2]


[48. Scraping3] – 데이터 베이스 저장
# data를 데이터 베이스에 넣기 전 csv에 넣는다면 오류 발생 없애기 위해 , 나 문제될만한 기호를 replace 해야!

nth가져오는 방법

[48. Scraping3]



[49. Scraping4]
# nth-child 를 쓸 수 있는 방법
nth-child(3)   nth-of-type(3)


[50. Scraping5]
.extend[]


[52.Scraping7_다음사이트에서별명가져오기]
# xPath

[54.서울시 구별 주유소 유가 정보 획득]
# 데이터프레임 붙이기
Concact -  station_raw = pd.concat(tmp_raw)

# swarmplot

[55. 머신러닝이란(개념)] 정리완료


[56. 머신러닝 첫걸음] 정리완료
# 데이터 학습 –  xor_data(문제)와 xor_label(답)을 함께 제시해서 학습
clf.fit(xor_data,xor_label)
#예측하기 – 패턴을 익혀서 답을 제시 
pre=clf.predict(xor_data)
# 정답률 구하기
ac_score = metrics.accuracy_score(xor_label, pre)
print('정답률 : ',ac_score) # 정답률 1.0 100퍼센트 모두 맞추었다.

[57. iris_KNN(분류)] 정리완료
# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(iris_dataset['data'],iris_dataset['target'], random_state = 0) #random_state = 0 난수를 한번 실행 후 고정
# 실제 훈련과 테스트를 해보기 이전에 신뢰할 수 있는 데이터인지 시각화를 통해 확인.
# random_state = 0 난수값 고정 랜덤
train_test_split(iris_dataset['data'],iris_dataset['target'], random_state = 0
# pip install mglearn
# plotting.scatter_matrix
pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize=(15,15), marker='o', s=20, alpha=0.8, cmap=mglearn.cm3) c는 컬럼, s는 점의 크기, cm3는 3종류의 색을 사용한다. 
# KNN 알고리즘
from sklearn.neighbors import KNeighborsClassifier #KNN분류기
knn = KNeighborsClassifier(n_neighbors = 1) # k의 값을 1로 정함.(근처에 1개만 걸려도 따라감)
# 훈련시키기
knn.fit(X_train, y_train)
# 데이터 예측하기
pre=knn.predict([[5.1, 3.0, 1.3, 0.2]])
print('예측 : {}'.format(pre))
print('예측이름 : {}'.format(iris_dataset['target_names'][pre]))
# 정답이 들어있는 y_test와 knn 모델이 예측한 y_pre의 일치도를 살펴보기 
print('테스트 셋의 정확도 : {:.2f}'.format(np.mean(y_pre == y_test )))


[58.iris(SVM, Vectior machine)을 이용한 분류] 정리완료
# Cross Validation(교차검증) 하기
clf = svm.SVC()
socores = model_selection.cross_val_score(clf, data,label,cv=3)

산포도에서 분리가 잘 되어있는 그림은 SVM이 잘 예측해주고, 
값들이 난잡한 것들은 RandomForest 가 좋다.
KNN은 교집합으로 되어 있을 때 잘 나옴. KNN 에서 값은 홀수로 주어야 한다. 그래야 구분이 됨 백인가 흑인가


[59. SVM을 활용한 비만도 (BMI) 측정 예측] 정리완료
# scatter 채워진 그래프
# 그래프 저장
plt.savefig('./data/bmi.png')


[60.독버섯 구분하기] 정리완료
#웹페이지에서 csv 저장
req.urlretrieve(url,local)
# 인덱스 생성 – 이뉴멀레이트와 다르게 한 개씩 다 가져온다.
Iterrows
# 데이터 내부의 기호를 숫자로 변환하기
#문자를 아스키코드값으로 출력
ord('A')
# 아스키코드값을 문자로 변환
chr(65)
# 리포트 생성
metrics.classification_report(test_label,pre)

[61. mLearning.png] 정리완료
# 이미지 사용
<img src = './data/mLearning.png'> 하고 마크다운사용하면 이미지 불러옴
# 위의 모든 머신러닝 툴을 불러오자
#-----------
from sklearn.neighbors import KNeighborsClassifier #Nearist Neighbors
from sklearn.svm import SVC # Linear SVM, RBF SVM
from sklearn.gaussian_process import GaussianProcessClassifier # Gaussian Process
from sklearnl.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # RRandomForest , AdaBoost
from sklearn.neural_network import  MLPClassifier #Neural Net
from sklearn.naive_bayes import  GaussianNB # Naive Bayes
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis #QDA
#-----------

[62 KONLPY(korea national laguage__) ] 정리완료
# sentences 는 문장을 만들어준다.
kkma.sentences
# 명사만 뽑아서 보여준다.
kn=kkma.nouns
# pos는 각 문법명을 명시해 준다.
kkma.pos
# Hannanum
# Okt

[63 영문 Word Cloud] 정리완료
# text 문서 불러서 word 분류
text = open('./data/Independence.txt').read()
wordcloud = WordCloud().generate(text)
# 빈도계산
wordcloud.words_
# font size를 지정하여 많은 단어를 보기
# word count를 지정하여 단어를 보기
# back ground 변경하기
# 해상도 변경!!! 표시되는 최대글자수 변경, 글자색 변경
# 만든 word cloud 파일로 받기
wordcloud.to_file('./data/wc_independence.png')
# 단색인 경우 옆의 칼라와 구분이 될 때 이렇게 숫자 데이터를 이용해서 워드클라우드를 원하는 모양으로 만들 수 있다.
# 흑백표시?
# 이미지 사용하여 워드클라우드 모양 만들기

[64 한글 word cloud] 정리완료
# 상위 50개 단어 확인
ko = nltk.Text(tokens_ko)
ko.vocab().most_common(50) # 상위 50개 단어를 횟수와 함께 가지고 온다.


