import pandas as pd
import numpy as np
import tensorflow as tf

#Step1:下载数据
#Step2:数据准备
ratings_df = pd.read_csv(r'F:\datasets\慕课电影推荐系统数据集\ml-latest-small\ratings.csv')
#使用tail查看后几行数据，默认5行
# ratings_df.tail()

movies_df = pd.read_csv(r'F:\datasets\慕课电影推荐系统数据集\ml-latest-small\movies.csv')
# movies_df.tail()

#增加行号信息
movies_df['movieRow'] = movies_df.index
# movies_df.tail()

##筛选movies_df中的特征
movies_df = movies_df[['movieRow', 'movieId', 'title']]
movies_df.to_csv('movieProcessed.csv', index=False, header=True, encoding='utf-8')
# movies_df.tail()

#将ratings_df中的movieId替换为行号
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
#使用head查看前几行数据，默认前5行
# ratings_df.head()

ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
ratings_df.to_csv('ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
# ratings_df.head()

#创建电影评分矩阵rating和评分记录矩阵record,即用户是否对电影进行了评分，评分则1，未评分则为0
userNo = ratings_df['userId'].max() + 1
movieNo = ratings_df['movieRow'].max() + 1

rating = np.zeros((movieNo,userNo))

flag = 0
ratings_df_length = np.shape(ratings_df)[0]  #ratings_df的样本个数

#填写rating
for index, row in ratings_df.iterrows():
    #将rating当中对应的电影编号及用户编号填上row当中的评分
    rating[int(row['movieRow']), int(row['userId'])] = row['rating']
    flag += 1
    print('processed %d, %d left' % (flag, ratings_df_length - flag))

#电影评分表中，>0表示已经评分，=0表示未被评分
record = rating > 0
#bool值转换为0和1
record = np.array(record, dtype=int)

##Step3:构建模型
#写一个函数，对评分取值范围进行缩放，这样能使评分系统性能更好一些
def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))  #初始化对于每部电影每个用户的平均评分
    rating_norm = np.zeros((m, n))  #保存处理后的数据
    #原始评分-平均评分，最后将计算结果和平均评分返回。
    for i in range(m):
        idx = record[i, :] != 0  #获取已经评分的电影的下标
        rating_mean[i] = np.mean(rating[i,  idx])  #计算平均值，右边式子代表第i行已经评分的电影的平均值
        rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
    return rating_norm, rating_mean

rating_norm, rating_mean = normalizeRatings(rating, record)

rating_norm = np.nan_to_num(rating_norm)
rating_mean = np.nan_to_num(rating_mean)

#假设有10中类型的电影
num_features = 10
#初始化电影矩阵X，用户喜好矩阵Theta,这里产生的参数都是随机数，并且是正态分布
X_parameters = tf.Variable(tf.random_normal([movieNo, num_features], stddev=0.35))
Theta_paramters = tf.Variable(tf.random_normal([userNo, num_features], stddev=0.35))
#理论课定义的代价函数
#tf.matmul(X_parameters, Theta_paramters, transpose_b=True)代表X_parameters和Theta_paramters的转置相乘
loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_paramters, transpose_b=True)
                             - rating_norm) * record) ** 2) \
       + 1/2 * (tf.reduce_sum(X_parameters**2)+tf.reduce_sum(Theta_paramters**2))  #正则化项，其中λ=1，可以调整来观察模型性能变化。

#创建优化器和优化目标
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

#Step4:训练模型
#使用TensorFlow中的tf.summary模块，它用于将TensorFlow的数据导出，从而变得可视化，因为loss是标量，所以使用scalar函数
tf.summary.scalar('loss', loss)
#将所有summary信息汇总
summaryMerged = tf.summary.merge_all()
#定义保存信息的路径
filename = 'F:\datasets\慕课电影推荐系统数据集\ml-latest-small\movie_tensorboard'
#把信息保存在文件当中
writer = tf.summary.FileWriter(filename)

#创建tensorflow绘画
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#开始训练模型
for i in range(5000):
    #不重要的变量可用下划线表示，每次train的结果都会保存到"_",summaryMerged的训练结果保存到"movie_summary"
    _, movie_summary = sess.run([train, summaryMerged])
    writer.add_summary(movie_summary, i)
    #记录一下次数，有没有都无所谓，只是看看还有多久
    print('i=', i, 'loss=', loss)
'''
进入cmd命令，进入文件位置后，输入"tensorboard --logdir=./ --host=127.0.0.1"
运行后，在浏览器输入127.0.0.1:6006,即可打开可视化TensorFlow的页面观察
'''
##Step5:评估模型
Current_X_paramters, Current_Theta_parameters = sess.run([X_parameters, Theta_paramters])
#将电影内容矩阵和用户喜好矩阵相乘，再加上每一行的均值，便得到一个完整的电影评分表
predicts = np.dot(Current_X_paramters, Current_Theta_parameters.T) + rating_mean
#计算预测值与真实值的残差平方和的算术平方根，将它作为误差error,随着迭代次数增加而减少
errors = np.sqrt(np.sum((predicts - rating)**2))

##Step6:构建完整的电影推荐系统
user_id = input('您要想哪位用户进行推荐？请输入用户编号：')
#获取对该用户的电影评分的列表，predicts[:, int(user_id)]是该用户对应的所有电影的评分，即系统预测的用户对于电影的评分
#argsort()从小到大排序，argsort()[::-1]从大到小排序
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
#idx用于保存已经推荐了多少部电影
idx = 0
print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))
for i in sortedResult:
    print('评分： %.2f, 电影名： %s' % (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
    idx += 1
    if idx == 20:
        break